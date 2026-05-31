import os
from copy import deepcopy
from typing import List, Optional

from omegaconf import flag_override
from typing_extensions import override

from fusion_bench import timeit_context
from fusion_bench.method.isotropic_merging.iso import (
    IsotropicMergingInCommonAndTaskSubspace,
    IsotropicMergingInCommonSubspace,
)
from fusion_bench.mixins import auto_register_config
from fusion_bench.modelpool import CausalLMBackbonePool, CausalLMPool
from fusion_bench.models.hf_utils import create_default_model_card
from fusion_bench.utils import instantiate
from fusion_bench.utils.pylogger import get_rankzero_logger

log = get_rankzero_logger(__name__)


@auto_register_config
class IsotropicMergingInCommonSubspaceForCausalLM(IsotropicMergingInCommonSubspace):
    """Iso-C (Isotropic Merging in Common Subspace) for CausalLM models.

    Uses CausalLMBackbonePool when `merge_backbone=True` so that `embed_tokens`
    and `lm_head` are preserved from the pretrained model (avoids the MergeBench
    vocab-expansion shape mismatch). When `merge_backbone=False`, pass
    `exclude_keys=['embed_tokens','lm_head']` in config to achieve the same.
    """

    def __init__(
        self,
        scaling_factor: float,
        exclude_keys: Optional[List[str]] = None,
        merge_backbone: bool = False,
        model_save_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            scaling_factor=scaling_factor,
            exclude_keys=exclude_keys,
        )

    @override
    def run(self, modelpool: CausalLMPool):
        if self.model_save_path is not None:
            tokenizer = modelpool.load_tokenizer()

        if self.merge_backbone:
            assert modelpool.has_pretrained
            log.info(
                "Merging backbone of the model pool, using CausalLMBackbonePool."
            )
            modelpool_config = deepcopy(modelpool.config)
            with flag_override(modelpool_config, "allow_objects", True):
                modelpool_config._target_ = (
                    "fusion_bench.modelpool.causal_lm.CausalLMBackbonePool"
                )
            backbone_modelpool = instantiate(modelpool_config)
            model = modelpool.load_model("_pretrained_")
            backbone_model = super().run(backbone_modelpool)
            model.model.layers = backbone_model
        else:
            model = super().run(modelpool)

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
                model_card_str = create_default_model_card(
                    models=[modelpool.get_model_path(m) for m in modelpool.model_names],
                    description=(
                        f"Merged with Iso-C (scaling_factor={self.scaling_factor})."
                    ),
                    algorithm_config=self.config,
                    modelpool_config=modelpool.config,
                )
                with open(os.path.join(self.model_save_path, "README.md"), "w") as f:
                    f.write(model_card_str)
        return model


@auto_register_config
class IsotropicMergingInCommonAndTaskSubspaceForCausalLM(
    IsotropicMergingInCommonAndTaskSubspace
):
    """Iso-CTS for CausalLM models. Same merge_backbone pattern as Iso-C."""

    def __init__(
        self,
        scaling_factor: float,
        common_space_fraction: float,
        exclude_keys: Optional[List[str]] = None,
        merge_backbone: bool = False,
        model_save_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            scaling_factor=scaling_factor,
            common_space_fraction=common_space_fraction,
            exclude_keys=exclude_keys,
        )

    @override
    def run(self, modelpool: CausalLMPool):
        if self.model_save_path is not None:
            tokenizer = modelpool.load_tokenizer()

        if self.merge_backbone:
            assert modelpool.has_pretrained
            log.info(
                "Merging backbone of the model pool, using CausalLMBackbonePool."
            )
            modelpool_config = deepcopy(modelpool.config)
            with flag_override(modelpool_config, "allow_objects", True):
                modelpool_config._target_ = (
                    "fusion_bench.modelpool.causal_lm.CausalLMBackbonePool"
                )
            backbone_modelpool = instantiate(modelpool_config)
            model = modelpool.load_model("_pretrained_")
            backbone_model = super().run(backbone_modelpool)
            model.model.layers = backbone_model
        else:
            model = super().run(modelpool)

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
                model_card_str = create_default_model_card(
                    models=[modelpool.get_model_path(m) for m in modelpool.model_names],
                    description=(
                        f"Merged with Iso-CTS "
                        f"(scaling_factor={self.scaling_factor}, "
                        f"common_space_fraction={self.common_space_fraction})."
                    ),
                    algorithm_config=self.config,
                    modelpool_config=modelpool.config,
                )
                with open(os.path.join(self.model_save_path, "README.md"), "w") as f:
                    f.write(model_card_str)
        return model
