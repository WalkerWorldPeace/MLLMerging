"""FusionBench wrapper for the WUDI / SWUDI / ASWUDI family."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin, auto_register_config

from .functional import dispatch_yongxianwei_merge
from .task_vector import get_param_names_to_merge

log = logging.getLogger(__name__)


@auto_register_config
class YongxianweiMergingAlgorithm(
    LightningFabricMixin,
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    """Unified FusionBench algorithm for the WUDI-family merging methods.

    The method to run is selected through ``method_name``:

    - ``wudi``   -- 300-step Adam on the per-layer quadratic interference proxy.
    - ``wudi2``  -- OptMerge: per-task SVD-truncated WUDI (ICLR 2026 baseline).
    - ``iwudi``  -- SWUDI-soft: closed-form soft spectral filter (full rank).
    - ``swudi``  -- SWUDI: soft filter + hard top-K spectral mask.
    - ``aswudi`` -- ASWUDI: SWUDI with per-layer zero-hyperparameter adaptive rank.

    Merging is applied only to floating-point parameters returned by
    ``named_parameters()`` whose name does not match
    ``exclude_param_names_regex``. 2-D weight matrices (excluding any
    ``lm_head.*``) are optimized with the method-specific objective; all
    other parameter shapes fall back to ``incremental_average``. Buffers
    and excluded parameters keep the pretrained values.
    """

    def __init__(
        self,
        method_name: str,
        scaling_coefficient: float = 1.0,
        exclude_param_names_regex: Optional[List[str]] = None,
        method_kwargs: Optional[Dict] = None,
        merge_device: Optional[str] = None,
        load_on_cpu: bool = False,
        strict: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_merge_device(self) -> torch.device:
        if self.merge_device is not None:
            return torch.device(self.merge_device)
        try:
            return self.fabric.device
        except Exception:  # pragma: no cover - fallback outside Fabric
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, modelpool: BaseModelPool, model_name: str) -> nn.Module:
        model = modelpool.load_model(model_name)
        if self.load_on_cpu:
            model = model.to("cpu")
        return model

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def run(self, modelpool: BaseModelPool) -> nn.Module:
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        if not modelpool.has_pretrained:
            raise ValueError(
                "YongxianweiMergingAlgorithm requires a '_pretrained_' model in the pool"
            )

        exclude_regex = list(self.exclude_param_names_regex or [])
        method_kwargs = dict(self.method_kwargs) if self.method_kwargs else {}

        # Inject expert_names so task-aware methods (e.g. swiglu_aware_aswudi)
        # can look up which task vector belongs to which expert. The
        # dispatcher's signature filter silently drops this for methods that
        # don't accept it.
        method_kwargs.setdefault("expert_names", list(modelpool.model_names))

        with self.profile("load pretrained"):
            pretrained_model = self._load_model(modelpool, "_pretrained_")

        pre_param_dict = {n: p for n, p in pretrained_model.named_parameters()}
        candidate_names = get_param_names_to_merge(
            list(pre_param_dict.keys()), exclude_regex
        )
        mergeable_names: List[str] = []
        finetuned_param_dicts: List[Dict[str, torch.Tensor]] = []

        for model_name in modelpool.model_names:
            with self.profile(f"load {model_name}"):
                finetuned_model = self._load_model(modelpool, model_name)
            fine_params: Dict[str, torch.Tensor] = {}
            for name, param in finetuned_model.named_parameters():
                fine_params[name] = param.detach()
            finetuned_param_dicts.append(fine_params)
            del finetuned_model

        # determine keys that are mergeable across all fine-tuned models
        for name in candidate_names:
            pre = pre_param_dict[name]
            if not pre.is_floating_point():
                continue
            ok = True
            for fine in finetuned_param_dicts:
                if name not in fine or fine[name].shape != pre.shape:
                    ok = False
                    break
            if ok:
                mergeable_names.append(name)

        skipped = sorted(set(candidate_names) - set(mergeable_names))
        if skipped:
            log.warning(
                "Skipping %d params not eligible for merging (shape/dtype/missing): %s",
                len(skipped), skipped[:10],
            )

        merge_device = self._resolve_merge_device()
        log.info(
            "yongxianwei_merging[%s]: merging %d params on %s "
            "(scaling_coefficient=%.4f, exclude_regex=%s)",
            self.method_name, len(mergeable_names), merge_device,
            self.scaling_coefficient, exclude_regex,
        )

        pre_subset = {n: pre_param_dict[n] for n in mergeable_names}
        finetuned_subsets = [
            {n: fine[n] for n in mergeable_names} for fine in finetuned_param_dicts
        ]
        del finetuned_param_dicts

        with self.profile(f"merge[{self.method_name}]"):
            merged_param_dict = dispatch_yongxianwei_merge(
                method_name=self.method_name,
                base_model=pre_subset,
                finetuned_models=finetuned_subsets,
                exclude_param_names_regex=None,  # already filtered
                scaling_coefficient=float(self.scaling_coefficient),
                method_kwargs=method_kwargs,
                merge_device=merge_device,
            )

        # Write merged params back onto a full state_dict baseline so buffers
        # and excluded parameters keep the pretrained values.
        full_state_dict = pretrained_model.state_dict()
        for name, tensor in merged_param_dict.items():
            target = full_state_dict.get(name)
            if target is None:
                continue
            full_state_dict[name] = tensor.to(dtype=target.dtype, device=target.device)

        missing, unexpected = pretrained_model.load_state_dict(
            full_state_dict, strict=self.strict
        )
        if missing:
            log.info("load_state_dict missing keys: %s", missing[:10])
        if unexpected:
            log.info("load_state_dict unexpected keys: %s", unexpected[:10])

        self.print_profile_summary()
        return pretrained_model
