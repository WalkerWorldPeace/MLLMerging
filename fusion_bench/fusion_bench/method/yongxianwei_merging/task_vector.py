"""TaskVector utilities migrated from ``model_merging.py``.

The ``TaskVector`` here mirrors the behaviour of the original script: it
computes the parameter-wise delta (``finetuned - pretrained``) for the
keys that are not filtered out by ``exclude_param_names_regex``. Buffers
are intentionally ignored because merging is defined only on parameters;
the wrapper algorithm re-uses the pretrained ``state_dict`` as the base
when writing merged weights back.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

import torch


def get_state_dict(model):
    """Extract a parameter dict from either a state dict or an ``nn.Module``.

    When ``model`` is an ``nn.Module`` we explicitly use ``named_parameters``
    so that buffers (e.g. BatchNorm running stats) never leak into the
    merge. Callers that want a full ``state_dict`` should call
    ``model.state_dict()`` directly.
    """
    if isinstance(model, dict):
        return model
    return {name: param for name, param in model.named_parameters()}


def get_param_names_to_merge(
    input_param_names: Iterable[str],
    exclude_param_names_regex: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return parameter names not matched by any of the exclusion regexes."""
    if exclude_param_names_regex is None:
        exclude_param_names_regex = []
    compiled = [re.compile(p) for p in exclude_param_names_regex]
    return [
        name
        for name in input_param_names
        if not any(pat.match(name) for pat in compiled)
    ]


def incremental_average(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Numerically stable running mean across a list of tensors."""
    result = tensors[0].clone()
    for i, t in enumerate(tensors[1:], 1):
        result = result + (t - result) / (i + 1)
    return result


class TaskVector:
    """Per-parameter delta ``finetuned - pretrained``.

    Compatible with the subset of the original ``model_merging.py`` API
    that the ported methods rely on: construction from two model-like
    objects (``nn.Module`` or ``dict``), construction from a pre-computed
    ``task_vector_param_dict``, addition, and reconstitution against a
    pretrained model.
    """

    def __init__(
        self,
        pretrained_model=None,
        finetuned_model=None,
        exclude_param_names_regex: Optional[Iterable[str]] = None,
        task_vector_param_dict: Optional[dict] = None,
    ):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
            return

        assert pretrained_model is not None and finetuned_model is not None, (
            "TaskVector requires either task_vector_param_dict or both "
            "pretrained_model and finetuned_model"
        )
        self.task_vector_param_dict = {}
        pretrained_param_dict = get_state_dict(pretrained_model)
        finetuned_param_dict = get_state_dict(finetuned_model)
        param_names_to_merge = get_param_names_to_merge(
            list(pretrained_param_dict.keys()), exclude_param_names_regex
        )

        with torch.no_grad():
            for name in param_names_to_merge:
                if name not in finetuned_param_dict:
                    continue
                pre = pretrained_param_dict[name]
                fine = finetuned_param_dict[name]
                if pre.shape != fine.shape:
                    continue
                if not pre.is_floating_point():
                    continue
                if pre.device != fine.device:
                    fine = fine.to(pre.device)
                self.task_vector_param_dict[name] = fine - pre

    def __add__(self, other: "TaskVector") -> "TaskVector":
        assert isinstance(other, TaskVector), (
            "addition of TaskVector can only be done with another TaskVector!"
        )
        new_dict = {}
        with torch.no_grad():
            for name in self.task_vector_param_dict:
                assert name in other.task_vector_param_dict, (
                    f"param_name {name} is not contained in both task vectors!"
                )
                new_dict[name] = (
                    self.task_vector_param_dict[name]
                    + other.task_vector_param_dict[name]
                )
        return TaskVector(task_vector_param_dict=new_dict)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def combine_with_pretrained_model(
        self, pretrained_model, scaling_coefficient: float = 1.0
    ) -> dict:
        pretrained_param_dict = get_state_dict(pretrained_model)
        merged = {}
        with torch.no_grad():
            for name, tv_param in self.task_vector_param_dict.items():
                pre = pretrained_param_dict[name]
                if pre.device != tv_param.device:
                    tv_param = tv_param.to(pre.device)
                merged[name] = pre + scaling_coefficient * tv_param
        return merged
