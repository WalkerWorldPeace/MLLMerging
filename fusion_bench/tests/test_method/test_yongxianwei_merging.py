"""Minimal unit tests for the WUDI / SWUDI / ASWUDI family.

These tests exercise the functional dispatcher and the
``YongxianweiMergingAlgorithm`` wrapper on toy state dicts. They do not
require CUDA and run in well under 30 s on CPU.

Coverage:
    * ``dispatch_yongxianwei_merge`` resolves all 5 registered methods
      and produces a state dict with the same keys, shapes, dtypes and
      finite values as the pretrained.
    * ``exclude_param_names_regex`` keeps matching parameters at their
      pretrained values.
    * ``YongxianweiMergingAlgorithm.run`` returns a module, preserves
      buffers, and moves the merged weights away from the pretrained.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fusion_bench.method.yongxianwei_merging.algorithm import (
    YongxianweiMergingAlgorithm,
)
from fusion_bench.method.yongxianwei_merging.functional import (
    _METHOD_REGISTRY,
    dispatch_yongxianwei_merge,
)
from fusion_bench.modelpool import BaseModelPool

# All five paper-line methods (mirrors functional._METHOD_REGISTRY).
METHODS = ["wudi", "wudi2", "iwudi", "swudi", "aswudi"]

# Per-method kwargs that keep the test cheap (≤ a handful of iterations).
_FAST_KWARGS = {
    "wudi":   {"iter_num": 1, "progress": False},
    "wudi2":  {"iter_num": 1, "progress": False},
    "iwudi":  {"progress": False, "landweber_steps": 5},
    "swudi":  {"progress": False},
    "aswudi": {"progress": False, "auto_rank_method": "entropy"},
}


def _make_toy_param_dicts(seed: int = 0):
    """Build a base + 3 fine-tuned dicts with mixed parameter shapes."""
    torch.manual_seed(seed)
    base = {
        "transformer.layer.weight": torch.randn(8, 16),  # 2-D, mergeable
        "transformer.layer.bias":   torch.randn(8),       # 1-D
        "vision.embed.weight":      torch.randn(2, 3, 4, 4),  # 4-D
        "layernorm.weight":         torch.ones(8),        # 1-D
    }
    finetuned = []
    for seed_i in range(3):
        torch.manual_seed(seed + seed_i + 1)
        delta = {k: 0.01 * torch.randn_like(v) for k, v in base.items()}
        finetuned.append({k: v + delta[k] for k, v in base.items()})
    return base, finetuned


class _ToyModel(nn.Module):
    """Tiny module with one mergeable Linear and one untouched buffer."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)
        self.register_buffer("running_stat", torch.zeros(4))


class TestRegistry(unittest.TestCase):
    def test_registry_keys(self):
        self.assertEqual(set(_METHOD_REGISTRY), set(METHODS))


class TestFunctionalDispatch(unittest.TestCase):
    def test_all_methods_preserve_keys_shapes_dtypes(self):
        base, finetuned = _make_toy_param_dicts()
        for method in METHODS:
            with self.subTest(method=method):
                merged = dispatch_yongxianwei_merge(
                    method_name=method,
                    base_model=base,
                    finetuned_models=finetuned,
                    exclude_param_names_regex=[],
                    scaling_coefficient=1.0,
                    method_kwargs=dict(_FAST_KWARGS[method]),
                )
                self.assertEqual(set(merged.keys()), set(base.keys()))
                for name, tensor in merged.items():
                    self.assertEqual(tensor.shape, base[name].shape)
                    self.assertEqual(tensor.dtype, base[name].dtype)
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_exclude_regex_keeps_pretrained(self):
        base, finetuned = _make_toy_param_dicts()
        merged = dispatch_yongxianwei_merge(
            method_name="wudi",
            base_model=base,
            finetuned_models=finetuned,
            exclude_param_names_regex=[r".*bias.*", r".*norm.*"],
            scaling_coefficient=1.0,
            method_kwargs=dict(_FAST_KWARGS["wudi"]),
        )
        # Excluded params either drop from the merged dict (algorithm wrapper
        # restores them from pretrained) or are returned unchanged. Both are
        # legitimate; what matters is they are NOT modified.
        for name in ("transformer.layer.bias", "layernorm.weight"):
            if name in merged:
                self.assertTrue(
                    torch.equal(merged[name], base[name]),
                    msg=f"{name} should be untouched but moved",
                )
        # Mergeable 2-D weight should have moved.
        moved = "transformer.layer.weight"
        self.assertIn(moved, merged)
        self.assertFalse(torch.equal(merged[moved], base[moved]))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            dispatch_yongxianwei_merge(
                method_name="not_a_method",
                base_model={},
                finetuned_models=[],
                exclude_param_names_regex=[],
                scaling_coefficient=1.0,
            )


class TestAlgorithmRun(unittest.TestCase):
    def _make_pool(self):
        torch.manual_seed(0)
        pretrained = _ToyModel()
        finetuned = []
        for i in range(2):
            m = _ToyModel()
            with torch.no_grad():
                for p_pre, p in zip(pretrained.parameters(), m.parameters()):
                    p.copy_(p_pre + 0.01 * torch.randn_like(p_pre))
                m.running_stat.copy_(torch.full_like(m.running_stat, float(i + 1)))
            finetuned.append(m)
        models = {"_pretrained_": pretrained}
        for i, m in enumerate(finetuned):
            models[f"t{i}"] = m
        return BaseModelPool(models), pretrained

    def test_run_returns_module_and_preserves_buffers(self):
        pool, pretrained = self._make_pool()
        pretrained_weight_snapshot = pretrained.linear.weight.detach().clone()
        algo = YongxianweiMergingAlgorithm(
            method_name="wudi",
            scaling_coefficient=1.0,
            exclude_param_names_regex=[],
            method_kwargs={"iter_num": 5, "progress": False},
            merge_device="cpu",
        )
        merged = algo.run(pool)
        self.assertIsInstance(merged, nn.Module)
        # Running buffer comes from the pretrained and is left at zero.
        self.assertTrue(torch.equal(merged.running_stat, torch.zeros(4)))
        # Weights should have moved compared to the pre-run snapshot.
        self.assertFalse(torch.equal(merged.linear.weight, pretrained_weight_snapshot))


if __name__ == "__main__":
    unittest.main()
