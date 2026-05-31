import logging
import os
from copy import deepcopy
from typing import List, Literal, Optional

from omegaconf import flag_override
from typing_extensions import override

from fusion_bench import timeit_context
from fusion_bench.method.dare.ties_merging import DareTiesMerging
from fusion_bench.mixins import auto_register_config
from fusion_bench.modelpool import CausalLMBackbonePool, CausalLMPool
from fusion_bench.models.hf_utils import create_default_model_card
from fusion_bench.utils import instantiate
from fusion_bench.utils.pylogger import get_rankzero_logger

log = get_rankzero_logger(__name__)


@auto_register_config
class DareTiesMergingForCausalLM(DareTiesMerging):
    """DARE + TIES merging adapted for CausalLM models.

    With `merge_backbone=True`, only transformer layers are merged via
    CausalLMBackbonePool; embeddings and lm_head are preserved from the
    pretrained model.
    """

    def __init__(
        self,
        sparsity_ratio: float,
        only_on_linear_weights: bool,
        rescale: bool,
        scaling_factor: float,
        threshold: int,
        remove_keys: Optional[List[str]] = None,
        merge_func: Literal["sum", "mean", "max"] = "sum",
        merge_backbone: bool = False,
        model_save_path: Optional[str] = None,
        **kwargs,
    ):
        if remove_keys is None:
            remove_keys = []
        super().__init__(
            sparsity_ratio=sparsity_ratio,
            only_on_linear_weights=only_on_linear_weights,
            rescale=rescale,
            scaling_factor=scaling_factor,
            threshold=threshold,
            remove_keys=remove_keys,
            merge_func=merge_func,
            **kwargs,
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
                        f"Merged with DARE + TIES "
                        f"(sparsity_ratio={self.sparsity_ratio}, "
                        f"scaling_factor={self.scaling_factor}, "
                        f"threshold={self.threshold})."
                    ),
                    algorithm_config=self.config,
                    modelpool_config=modelpool.config,
                )
                with open(os.path.join(self.model_save_path, "README.md"), "w") as f:
                    f.write(model_card_str)
        return model
