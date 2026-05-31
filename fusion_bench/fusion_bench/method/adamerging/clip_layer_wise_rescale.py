"""
SWUDI (or any single merged task vector) layer-wise test-time rescaling.

This is the dual to standard layer-wise AdaMerging:
    standard:  W = W_base + Σ_i λ_{i,l} * (W_finetuned_i - W_base)
    rescale:   W = W_base + beta_l * (W_merged - W_base)

with `W_merged` an externally-merged checkpoint (e.g. SWUDI). `beta_l` is
initialized to 1.0 so the model starts strictly equal to `W_merged`; only
`beta_l` is optimized via test-time entropy minimization on CLIP test images.

This is used as a control experiment to test whether AdaMerging's gain over
SWUDI comes from a global per-layer rescaling of the merged direction, or
whether it requires the per-task residual structure that standard AdaMerging
exposes.

Example Usage:

```bash
fusion_bench seed=42 \
    method=adamerging/swudi_layer_wise_rescale \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_swudi_rescale \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    report_save_path=outputs/.../swudi_layer_rescale.json
```
"""

import functools
import logging
import os
from typing import TYPE_CHECKING, List, Optional

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.models.wrappers.layer_wise_rescale import LayerWiseRescaleModel
from fusion_bench.utils.data import load_tensor_from_file
from fusion_bench.utils.type import TorchModelType

from .entropy_loss import entropy_loss
from .utils import get_memory_usage

if TYPE_CHECKING:
    from fusion_bench.programs.fabric_fusion_program import FabricModelFusionProgram

log = logging.getLogger(__name__)


class CLIPLayerWiseRescaleAlgorithm(
    CLIPClassificationMixin,
    LightningFabricMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    """Test-time layer-wise rescaling of a single merged CLIP vision encoder."""

    _program: "FabricModelFusionProgram"

    def __init__(self, **kwargs):
        # Store all YAML kwargs in a DictConfig so the rest of this class can
        # use the legacy `self.config.<key>` accessor pattern (mirroring
        # CLIPLayerWiseAdaMergingAlgorithm).
        algorithm_config = OmegaConf.create(dict(kwargs))
        super().__init__(algorithm_config)

    # ---- modelpool helpers ----------------------------------------------------

    def _get_merged_model_name(self, modelpool: "ModelPool") -> str:
        """Pick the model name treated as `W_merged`.

        - If the config explicitly sets `merged_model_name`, use it.
        - Else expect exactly one non-pretrained model in the pool.
        """
        explicit = self.config.get("merged_model_name", None)
        if explicit is not None:
            return explicit
        names = list(modelpool.model_names)
        if len(names) != 1:
            raise ValueError(
                "swudi_layer_wise_rescale expects exactly one non-pretrained model "
                f"in the modelpool (the merged checkpoint), got {names}. "
                "Either trim the modelpool or set `method.merged_model_name=...`."
            )
        return names[0]

    def _get_task_names(self, modelpool: "ModelPool") -> List[str]:
        """Task names for entropy minimization. Defaults to test_dataset_names."""
        names = self.config.get("task_names", None)
        if names is not None:
            return list(names)
        names = list(modelpool.test_dataset_names)
        if not names:
            raise ValueError(
                "Cannot determine task names: no test_datasets in the modelpool "
                "and no `method.task_names` set."
            )
        return names

    @torch.no_grad()
    def construct_rescale_model(
        self, modelpool: "ModelPool"
    ) -> LayerWiseRescaleModel:
        pretrained_model = modelpool.load_model("_pretrained_")
        merged_name = self._get_merged_model_name(modelpool)
        merged_model = modelpool.load_model(merged_name)

        init_value = float(self.config.get("init_value", 1.0))
        module = LayerWiseRescaleModel(
            pretrained_model=pretrained_model,
            merged_model=merged_model,
            init_value=init_value,
            clamp_weights=self.config.get("clamp_weights", False),
            tie_weights=self.config.get("tie_weights", True),
            strict=self.config.get("strict", False),
        )

        # Optional: pre-load beta from file (skip TTA).
        weights = self.config.get("weights", None)
        if isinstance(weights, str) and weights:
            loaded = load_tensor_from_file(weights)
            if loaded.shape != module.merge_weight.shape:
                raise ValueError(
                    f"Loaded beta shape {loaded.shape} does not match expected "
                    f"{module.merge_weight.shape}"
                )
            with torch.no_grad():
                module.merge_weight.copy_(loaded.to(module.merge_weight))

        log.info(
            f"LayerWiseRescaleModel: num_layers={module.merge_weight.numel()}, "
            f"init_value={init_value}, merged_model_name={merged_name}"
        )
        return module

    # ---- CLIP-specific hooks --------------------------------------------------

    def on_test_time_adaptation_start(self):
        # Make sure zero-shot heads are built for ALL task names (test datasets),
        # not just `modelpool.model_names` (which is just the merged checkpoint).
        self.setup_zero_shot_classification_head(task_names=self._task_names)

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str):
        return CLIPClassificationMixin.get_shuffled_test_loader_iter(
            self,
            task,
            batch_size=self.config.get("batch_size", 16),
            num_workers=self.config.get("num_workers", 8),
        )

    # ---- saving ---------------------------------------------------------------

    @rank_zero_only
    def save_merging_weights(self, file_path: str, merging_weights: torch.Tensor):
        if not (
            self.fabric.is_global_zero
            and self.config.get("save_merging_weights", False)
        ):
            return
        if isinstance(file_path, str) and not file_path.startswith(("/", ".")):
            save_path = os.path.join(self.log_dir, file_path)
        else:
            save_path = file_path
        log.info(f"saving learned beta to {save_path}.")
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(merging_weights.detach().cpu(), save_path)

    # ---- main entry -----------------------------------------------------------

    def run(self, modelpool: ModelPool, **kwargs):
        log.info(
            "Fusing models using single-task-vector layer-wise rescaling "
            "(test-time entropy minimization on `W_base + beta_l * Delta`)."
        )
        self.modelpool = modelpool
        self._task_names = self._get_task_names(modelpool)
        self.log_hyperparams(self.config)

        with self.profile("construct the wrapped model"):
            module = self.construct_rescale_model(modelpool)

        if (
            isinstance(self.config.get("weights", None), str)
            and self.config.get("weights")
        ):
            return module.merge_and_unload()

        with self.profile("test-time adaptation"):
            module = self.test_time_adaptation(module)
        if self.config.get("save_merging_weights", False):
            self.save_merging_weights(
                self.config.save_merging_weights, module.merge_weight
            )
        # Print final beta statistics.
        self._log_beta_stats(module.merge_weight.detach())
        return module.merge_and_unload()

    @rank_zero_only
    def _log_beta_stats(self, beta: torch.Tensor):
        b = beta.detach().cpu().float().flatten()
        eps = float(self.config.get("beta_one_eps", 1e-3))
        deviating = ((b - 1.0).abs() > eps).sum().item()
        log.info(
            "[swudi_layer_rescale] learned beta: "
            f"n={b.numel()}, min={b.min().item():.4f}, max={b.max().item():.4f}, "
            f"mean={b.mean().item():.4f}, std={b.std().item():.4f}; "
            f"|beta_l - 1| > {eps}: {deviating}/{b.numel()} "
            f"({deviating / max(b.numel(), 1) * 100:.1f}%)"
        )

    # ---- TTA loop -------------------------------------------------------------

    def test_time_adaptation(
        self, module: "LayerWiseRescaleModel[TorchModelType]"
    ) -> "LayerWiseRescaleModel[TorchModelType]":
        self.on_test_time_adaptation_start()

        opt_name = self.config.get("optimizer", "adam").lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                [module.merge_weight], lr=float(self.config.lr)
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                [module.merge_weight],
                lr=float(self.config.lr),
                momentum=float(self.config.get("momentum", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        log.info(f"{optimizer=}")
        module, optimizer = self.fabric.setup(module, optimizer)

        module.train()
        module.merge_weights()

        max_steps = self.config.max_steps if not self.is_debug_mode else 1
        for step_idx in (
            pbar := tqdm(
                range(max_steps),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "SWUDI layer-wise rescale TTA",
                dynamic_ncols=True,
            )
        ):
            for task in self._task_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, batch[0], task)
                    loss = entropy_loss(logits)
                with self.profile("backward pass"):
                    self.fabric.backward(loss, retain_graph=True)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()
            with self.profile("merging weights"):
                module.merge_weights()

            beta = module.merge_weight.detach()
            metrics = {
                "train/loss": float(loss.item()),
                "train/beta_max": float(beta.max().item()),
                "train/beta_min": float(beta.min().item()),
                "train/beta_mean": float(beta.mean().item()),
                "train/beta_std": float(beta.std().item()),
            }
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

        log.info(get_memory_usage("after swudi layer-wise rescale, GPU memory:"))
        self.print_profile_summary()
        return module
