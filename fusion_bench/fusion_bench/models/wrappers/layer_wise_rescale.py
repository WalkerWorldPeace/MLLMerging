"""
Layer-wise rescaling wrapper for a single merged task vector.

Given a base model `W_base` and a single merged checkpoint `W_merged` (e.g. the
output of SWUDI), this wrapper exposes a learnable per-parameter scalar
`beta_l` and performs the forward pass with:

    W_l = W_base,l + beta_l * (W_merged,l - W_base,l)

`beta_l` is initialized to 1.0, so the wrapped model starts strictly equal to
`W_merged`. Only `beta_l` is trainable; both `W_base` and `Delta = W_merged -
W_base` are frozen.
"""

import functools
from copy import deepcopy
from typing import Generic, Optional

import torch
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.models.utils import del_attr, get_attr
from fusion_bench.utils.type import StateDictType, TorchModelType

__all__ = ["LayerWiseRescaleModel"]


class LayerWiseRescaleModel(nn.Module, Generic[TorchModelType]):
    """Wraps `W_base + beta_l * Delta` for a single merged task vector.

    The wrapper iterates over the parameters of `pretrained_model` whose
    `requires_grad` is True; for each such parameter it learns a single scalar
    `beta_l`.  Non-trainable parameters of the pretrained model are kept frozen
    and shared.
    """

    _merged_state_dict: Optional[StateDictType] = None

    def __init__(
        self,
        pretrained_model: TorchModelType,
        merged_model: TorchModelType,
        init_value: float = 1.0,
        clamp_weights: bool = False,
        tie_weights: bool = True,
        strict: bool = False,
    ):
        super().__init__()
        self.clamp_weights = clamp_weights
        self.tie_weights = tie_weights
        self.strict = strict

        # Drop any non-trainable params from the merged copy so we don't carry
        # duplicate buffers around.
        for name, param in pretrained_model.named_parameters():
            if not param.requires_grad:
                del_attr(merged_model, name.split("."))
            else:
                merged_attr = get_attr(merged_model, name.split("."))
                # Convert merged_model in-place into the delta `W_merged - W_base`.
                merged_attr.data = merged_attr.data - param.data

        self.pretrained_model = pretrained_model.requires_grad_(False)
        merged_model.requires_grad_(False)
        # `delta_model` now stores the per-parameter task vector.
        self.delta_model = merged_model

        num_layers = sum(
            1 for _ in self.delta_model.parameters()
        )
        # Shape `(num_layers,)` — one scalar per trainable parameter.
        beta = torch.full(
            (num_layers,),
            float(init_value),
            dtype=next(self.delta_model.parameters()).dtype
            if num_layers > 0
            else torch.float32,
        )
        self.merge_weight = nn.Parameter(beta, requires_grad=True)

    @property
    def forward_model(self):
        return functools.partial(
            functional_call,
            self.pretrained_model,
            self._merged_state_dict,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )

    def merge_weights(self) -> StateDictType:
        if self.clamp_weights:
            beta = self.merge_weight.clamp(0, 1)
        else:
            beta = self.merge_weight

        state_dict = self.pretrained_model.state_dict(keep_vars=True)
        for w, (name, delta_param) in zip(beta, self.delta_model.named_parameters()):
            state_dict[name] = state_dict[name] + delta_param * w
        self._merged_state_dict = state_dict
        return state_dict

    def merge_and_unload(self, copy: bool = False) -> TorchModelType:
        self.merge_weights()
        if copy:
            model = deepcopy(self.pretrained_model)
        else:
            model = self.pretrained_model
        model.load_state_dict(self._merged_state_dict)
        return model

    def forward(self, *args, **kwargs):
        if self._merged_state_dict is None:
            self.merge_weights()
        return self.forward_model(args=args, kwargs=kwargs)
