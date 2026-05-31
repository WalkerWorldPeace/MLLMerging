"""WUDI / SWUDI / ASWUDI family for FusionBench.

This package implements the spectral-regularization model-merging methods
introduced in the manuscript *"Spectral Regularization for Data-Free Model
Merging"*:

- ``wudi``  -- WUDI: 300-step Adam on the per-layer quadratic interference proxy.
- ``iwudi`` -- SWUDI-soft: closed-form soft spectral filter (full rank).
- ``swudi`` -- SWUDI: soft filter + hard top-K spectral mask.
- ``aswudi`` -- ASWUDI: SWUDI with per-layer zero-hyperparameter adaptive rank.

A reference re-implementation of OptMerge (per-task SVD-truncated WUDI,
``method_name="wudi2"``) is also kept for baseline comparison.

See the top-level ``README.md`` for the unified filter formula, main
results, and reproduction commands.
"""

from .algorithm import YongxianweiMergingAlgorithm

__all__ = ["YongxianweiMergingAlgorithm"]
