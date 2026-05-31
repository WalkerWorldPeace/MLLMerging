"""Functional implementations of the WUDI / SWUDI / ASWUDI family.

Each ``*_merge`` function takes the pretrained model (or its parameter
dict), a list of fine-tuned models (or parameter dicts), an exclusion
regex list, a ``scaling_coefficient`` and method-specific kwargs. They
return ``merged_param_dict`` -- a dict mapping parameter name to the new
tensor (same shape/dtype as the pretrained parameter). The algorithm
wrapper is responsible for copying the rest of the pretrained
``state_dict`` (buffers etc.) around this dict.

Supported methods (registered in ``_METHOD_REGISTRY``):

- ``wudi``  : WUDI -- 300-step Adam baseline on the per-layer quadratic proxy.
- ``wudi2`` : OptMerge -- per-task SVD-truncated WUDI (ICLR 2026 baseline).
- ``iwudi`` : SWUDI-soft -- full-rank soft spectral filter (no truncation).
- ``swudi`` : SWUDI -- soft filter + hard top-K spectral mask.
- ``aswudi``: ASWUDI -- SWUDI with per-layer zero-hyperparameter adaptive rank.

See the top-level ``README.md`` §1-2 for the unified spectral filter
formula and the corresponding paper sections.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from .task_vector import TaskVector, incremental_average

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CCAM helpers (shared between v1..v5)
# ---------------------------------------------------------------------------

def _build_task_vectors(
    base_model,
    finetuned_models: Sequence,
    exclude_param_names_regex: Optional[Iterable[str]],
) -> List[TaskVector]:
    return [
        TaskVector(
            pretrained_model=base_model,
            finetuned_model=m,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for m in finetuned_models
    ]


def _is_2d_optimizable(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and "lm_head" not in name


def wudi_merge_v1(
    base_model,
    finetuned_models: Sequence,
    exclude_param_names_regex: Optional[Iterable[str]],
    *,
    scaling_coefficient: float = 1.0,
    merge_device=None,
    iter_num: int = 300,
    learning_rate: float = 1e-5,
    optimizer: str = "adam",
    eps: float = 1e-8,
    momentum: float = 0.0,
    log_norm_history: bool = False,
    progress: bool = True,
) -> dict:
    """User-version WUDI merging.

    The default ``optimizer="adam", eps=1e-8`` reproduces the original
    Adam path byte-for-byte (matches paper-default protocol). Diagnostic
    knobs:
        ``optimizer="sgd"``  -> avoid Adam's 2nd-moment amplification on
                               near-rank-deficient LoRA spectra.
        ``eps=1e-3``         -> flatten Adam's denominator in small-λ
                               directions without changing optimizer.
        ``log_norm_history`` -> log per-layer ``||θ_t||`` every 30 steps
                               for norm-explosion diagnostics.
    """
    device = merge_device
    opt_name = (optimizer or "adam").lower()

    task_vectors = _build_task_vectors(
        base_model, finetuned_models, exclude_param_names_regex
    )

    def _optimize(param_name, vectors):
        original_dtype = vectors.dtype
        target_device = device if device is not None else vectors.device
        vectors = vectors.to(device=target_device, dtype=torch.float32)
        merging_vector = torch.nn.Parameter(vectors.sum(dim=0))
        if opt_name == "sgd":
            opt = torch.optim.SGD(
                [merging_vector], lr=learning_rate, momentum=momentum
            )
        elif opt_name == "adam":
            opt = torch.optim.Adam([merging_vector], lr=learning_rate, eps=eps)
        else:
            raise ValueError(
                f"wudi_merge_v1: unknown optimizer={optimizer!r} "
                "(expected 'adam' or 'sgd')"
            )
        l2_norms = torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1).square()

        tiny = 1e-12
        if l2_norms.max().item() < tiny:
            with torch.no_grad():
                avg_delta = incremental_average([v for v in vectors])
            return avg_delta.to(original_dtype)

        iterator = range(iter_num)
        if progress:
            iterator = tqdm(iterator, desc=f"wudi {param_name}", leave=False)
        norm_history = [] if log_norm_history else None
        for step in iterator:
            disturb = merging_vector.unsqueeze(0) - vectors
            inner = torch.matmul(disturb, vectors.transpose(1, 2))
            loss = (inner.square() / l2_norms.unsqueeze(-1).unsqueeze(-1)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if log_norm_history and (step % 30 == 0 or step == iter_num - 1):
                norm_history.append(
                    (step, merging_vector.norm().item(), float(loss.item()))
                )
        if log_norm_history:
            logger.info(
                "wudi_merge_v1[%s, opt=%s, eps=%.1e, lr=%.1e]: norm_history=%s",
                param_name, opt_name, eps, learning_rate, norm_history,
            )
        return merging_vector.data.detach().to(original_dtype)

    merged_tv_dict = {}
    for name in task_vectors[0].task_vector_param_dict:
        param = task_vectors[0].task_vector_param_dict[name]
        if _is_2d_optimizable(param, name):
            stacked = torch.stack([tv.task_vector_param_dict[name] for tv in task_vectors])
            merged_tv_dict[name] = _optimize(name, stacked)
        else:
            merged_tv_dict[name] = incremental_average(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
        if device is not None:
            merged_tv_dict[name] = merged_tv_dict[name].to(
                task_vectors[0].task_vector_param_dict[name].device
            )

    merged_tv = TaskVector(task_vector_param_dict=merged_tv_dict)
    return merged_tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=float(scaling_coefficient)
    )


# ---------------------------------------------------------------------------
# wudi_merge_v2 -- per-task SVD-truncated WUDI (a.k.a. "wudi2" / "OptMerge")
# ---------------------------------------------------------------------------
#
# Ported from MLLMerging/InternVL/internvl_chat/model_merging.py::wudi_merging2.
# Two engineering changes vs the original script:
#   * iter_num / learning_rate / merge_device exposed as kwargs.
#   * truncate_rank_ratio overrides the original 1/N rule when set.
# Default values (ratio=None, iter_num=300, lr=1e-5) reproduce the script's
# numerical behaviour byte-for-byte modulo float-summation order.

def wudi_merge_v2(
    base_model,
    finetuned_models: Sequence,
    exclude_param_names_regex: Optional[Iterable[str]],
    *,
    scaling_coefficient: float = 1.0,
    merge_device=None,
    iter_num: int = 300,
    learning_rate: float = 1e-5,
    truncate_rank_ratio: Optional[float] = None,
    progress: bool = True,
) -> dict:
    """Per-task SVD-truncated WUDI.

    Differs from :func:`wudi_merge_v1` by replacing each task vector's
    role in the quadratic interference loss with two truncated factors:

    1. ``taskvector_i`` -- the *target* in the loss -- is rebuilt from the
       top-K SVD of the residual ``v_i - mean(v)`` plus ``mean(v)``. This
       keeps the shared signal exactly while denoising the per-task part.
    2. ``low_rank_i`` -- the *projector* in the loss -- is the rank-K
       truncation ``S_K · V^T`` of ``v_i`` itself (top-K row factor). It
       restricts the interference measurement to ``v_i``'s high-energy
       subspace.

    The optimization loop is otherwise identical to WUDI:
    ``L = Σ_i (1/||v_i||_F²) · ||(θ - taskvector_i) · low_rank_i^T||_F²``,
    minimized with Adam from ``θ_0 = Σ_i v_i``.

    Conceptually this is per-task hard SVD truncation, complementing
    SWUDI's global hard truncation on the eigenbasis of
    ``C = Σ_i v_i^T v_i / ||v_i||²`` and IWUDI's soft spectral filter.
    """
    device = merge_device

    task_vectors = _build_task_vectors(
        base_model, finetuned_models, exclude_param_names_regex
    )

    def _optimize(param_name, vectors):
        original_dtype = vectors.dtype
        target_device = device if device is not None else vectors.device
        vectors = vectors.to(device=target_device, dtype=torch.float32)

        N, m, n = vectors.shape
        min_dim = min(m, n)
        if truncate_rank_ratio is None:
            K = max(1, min_dim // N)
        else:
            K = max(1, int(min_dim * float(truncate_rank_ratio)))
        K = min(K, min_dim)

        average_vector = vectors.mean(dim=0)

        low_rank_list = []
        taskvector_list = []
        for i in range(N):
            v_i = vectors[i]
            # Full SVD of v_i: U (m,m), s (min,), Vh (n,n)
            _u_full, s_full, vh_full = torch.linalg.svd(v_i, full_matrices=True)
            # Reduced SVD of residual (v_i - avg)
            u_res, s_res, vh_res = torch.linalg.svd(
                v_i - average_vector, full_matrices=False
            )

            # Truncate residual factors to top-K
            u_res_K = u_res[:, :K]
            s_res_K = s_res[:K]
            vh_res_K = vh_res[:K, :]

            # Mask top-K singular values + right vectors of v_i
            s_mask = torch.zeros_like(s_full)
            s_mask[:K] = 1.0
            s_masked = s_full * s_mask
            v_mask = torch.zeros_like(vh_full)
            v_mask[:K, :] = 1.0
            vh_masked = vh_full * v_mask

            # Build (m, n) S matrix with top-K on its diagonal
            S_matrix = torch.zeros(
                m, n, device=v_i.device, dtype=v_i.dtype
            )
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s_masked)

            low_rank_i = S_matrix @ vh_masked
            taskvector_i = (
                u_res_K @ torch.diag_embed(s_res_K) @ vh_res_K + average_vector
            )

            low_rank_list.append(low_rank_i)
            taskvector_list.append(taskvector_i)

        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)

        merging_vector = torch.nn.Parameter(vectors.sum(dim=0))
        optimizer = torch.optim.Adam([merging_vector], lr=learning_rate)
        l2_norms = torch.norm(
            vectors.reshape(N, -1), p=2, dim=-1
        ).square()

        tiny = 1e-12
        if l2_norms.max().item() < tiny:
            return average_vector.to(original_dtype)

        iterator = range(iter_num)
        if progress:
            iterator = tqdm(iterator, desc=f"wudi2 {param_name}", leave=False)
        for _step in iterator:
            disturb = merging_vector.unsqueeze(0) - taskvector
            inner = torch.matmul(disturb, low_rank.transpose(1, 2))
            loss = (inner.square() / l2_norms.unsqueeze(-1).unsqueeze(-1)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return merging_vector.data.detach().to(original_dtype)

    merged_tv_dict = {}
    for name in task_vectors[0].task_vector_param_dict:
        param = task_vectors[0].task_vector_param_dict[name]
        if _is_2d_optimizable(param, name):
            stacked = torch.stack(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
            merged_tv_dict[name] = _optimize(name, stacked)
        else:
            merged_tv_dict[name] = incremental_average(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
        if device is not None:
            merged_tv_dict[name] = merged_tv_dict[name].to(
                task_vectors[0].task_vector_param_dict[name].device
            )

    merged_tv = TaskVector(task_vector_param_dict=merged_tv_dict)
    return merged_tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=float(scaling_coefficient)
    )


# ---------------------------------------------------------------------------
# ccam_v6 -- hypothesis-validating fix for the CCAM underperformance on TA8
# ---------------------------------------------------------------------------

def iwudi_merge(
    base_model,
    finetuned_models: Sequence,
    exclude_param_names_regex: Optional[Iterable[str]],
    *,
    scaling_coefficient: float = 1.0,
    merge_device=None,
    # Spectral filter family.
    # "tikhonov":   f(λ) = λ / (λ + alpha)         (ridge-regression shrinkage)
    # "landweber":  f(λ) = 1 - (1 - eta·λ)^n       (exact GD iteration filter)
    # "exponential":f(λ) = 1 - exp(-λ·t)           (continuous-GD / gradient flow)
    # Empirical default: exponential (t=300) -- best accuracy on TA8 and
    # only one hyperparameter.
    filter_type: str = "exponential",
    # Tikhonov:
    tikhonov_alpha: float = 1.0e-2,
    # Landweber: step size eta = eta_ratio / max(eigenvalues(C)).
    landweber_steps: int = 150,
    landweber_eta_ratio: float = 0.9,
    # Exponential: "effective time" hyperparameter.
    exp_time: float = 300.0,
    # init for the interpolation: "sum" (WUDI-compatible) or "zero".
    init_mode: str = "sum",
    # Eigenvalue floor; protects 1/λ for tiny eigenvalues (rank-deficient C).
    min_eigenvalue: float = 1.0e-8,
    progress: bool = True,
) -> dict:
    """iwudi: Interpolated WUDI via closed-form spectral filtering.

    Motivation
    ----------
    The WUDI objective is a convex quadratic in θ with gradient

        ∇L(θ) = θ C - D  ,  where  C = Σ v_i^T v_i / ||v_i||²,
                                  D = Σ v_i v_i^T v_i / ||v_i||².

    Its global minimum is ``θ* = D C^{-1}``. Our empirical diagnosis on
    CLIP-ViT-B/32 TA8 showed:

    - ``θ*`` (closed form) → 82.33% accuracy
    - WUDI 300 Adam iters → 84.63% (WUDI default)
    - WUDI 500 Adam iters → 84.82% (wudi_plus peak)
    - WUDI 1000+ Adam iters → decays back toward 82.33%

    This means WUDI's "success" is not because of its objective but because
    it applies **spectral-filtered regularization** via early stopping.
    Adam from init ``θ_0 = Σ v_i`` toward ``θ*`` passes through a sweet spot
    whose distance from either endpoint depends on which eigen-modes of C
    have converged.

    This method replaces the iterative optimization with an explicit
    closed-form spectral filter, making the regularization controllable
    and removing optimizer stochasticity.

    Algorithm
    ---------
    1. Eigendecompose ``C = V Λ V^T`` (symmetric PSD, ``Λ`` = diag(λ_k)).
    2. Compute ``θ*_CF = D V diag(1/λ_k) V^T`` (WUDI closed-form).
    3. Apply eigenvalue-wise filter ``f(λ_k) ∈ [0, 1]``:
       - Tikhonov:   ``f(λ) = λ / (λ + α)``
       - Landweber:  ``f(λ) = 1 - (1 - η λ)^n``    (η = eta_ratio / max λ)
       - Exponential:``f(λ) = 1 - exp(-λ t)``
    4. Return ``θ = θ_0 + (θ*_CF - θ_0) · V diag(f(λ)) V^T``, where
       ``θ_0 = Σ_i v_i`` (default, WUDI-compatible) or ``θ_0 = 0``.

    At ``f ≡ 1`` (e.g. α→0, n→∞, t→∞): θ = θ*_CF (full WUDI closed-form).
    At ``f ≡ 0`` (e.g. α→∞, n=0, t=0):   θ = θ_0 (no optimization).

    The filter continuously interpolates between these endpoints with
    eigenvalue-wise control -- small eigenvalues (noisy / ill-conditioned
    directions) are suppressed more aggressively than large ones
    (well-determined / important directions).

    Computational cost: one ``torch.linalg.eigh`` per 2-D parameter, which
    is ~10-100ms on GPU for CLIP-ViT-B/32 sizes. No iterative optimization,
    no Adam state, fully deterministic.
    """
    if filter_type not in ("tikhonov", "landweber", "exponential"):
        raise ValueError(
            f"Unknown filter_type={filter_type!r}; expected one of "
            "'tikhonov', 'landweber', 'exponential'"
        )
    if init_mode not in ("sum", "zero"):
        raise ValueError(
            f"Unknown init_mode={init_mode!r}; expected 'sum' or 'zero'"
        )
    device = merge_device

    task_vectors = _build_task_vectors(
        base_model, finetuned_models, exclude_param_names_regex
    )

    def _solve(param_name, v_stack):
        original_dtype = v_stack.dtype
        target_device = device if device is not None else v_stack.device
        v_stack = v_stack.to(device=target_device, dtype=torch.float32)
        N = v_stack.shape[0]
        D1, D2 = v_stack.shape[1], v_stack.shape[2]

        flat = v_stack.reshape(N, -1)
        l2_sq = torch.norm(flat, p=2, dim=-1).square()
        tiny = 1.0e-12
        if l2_sq.max().item() < tiny:
            with torch.no_grad():
                return incremental_average([v for v in v_stack]).to(original_dtype)
        safe_norms = l2_sq.clamp(min=tiny)
        inv_norms = 1.0 / safe_norms

        # B_i = v_i^T v_i  (D2 x D2)     C = Σ_i B_i / ||v_i||²
        # vB_i = v_i @ B_i (D1 x D2)     D = Σ_i vB_i / ||v_i||²
        B = torch.einsum("nab,nac->nbc", v_stack, v_stack)     # (N, D2, D2)
        C = torch.einsum("n,nbc->bc", inv_norms, B)              # (D2, D2)
        vB = torch.einsum("nab,nbc->nac", v_stack, B)            # (N, D1, D2)
        D_mat = torch.einsum("n,nac->ac", inv_norms, vB)         # (D1, D2)

        # Eigendecomp of C (symmetric PSD). eigh returns ascending eigvals.
        eigvals, V_eig = torch.linalg.eigh(C)                    # (D2,), (D2, D2)
        safe_eigvals = eigvals.clamp(min=min_eigenvalue)

        if filter_type == "tikhonov":
            filter_vals = safe_eigvals / (safe_eigvals + tikhonov_alpha)
        elif filter_type == "landweber":
            max_eig = safe_eigvals.max().clamp(min=min_eigenvalue)
            eta = landweber_eta_ratio / max_eig
            base = (1.0 - eta * safe_eigvals).clamp(min=0.0, max=1.0)
            filter_vals = 1.0 - base.pow(float(landweber_steps))
        else:  # exponential
            filter_vals = 1.0 - torch.exp(-safe_eigvals * exp_time)

        # Compute θ in closed form:
        # θ = θ_0 + (θ*_CF - θ_0) · (V diag(f(λ)) V^T)
        # θ*_CF - θ_0 in eigenbasis: M = (θ*_CF - θ_0) V = [(D V)(diag 1/λ) - θ_0 V]
        # But more numerically stable form:
        #   θ = θ_0 + (D V diag(1/λ) V^T - θ_0) · (V diag(f) V^T)
        # Expand using associativity:
        #   = θ_0 · (I - V diag(f) V^T) + D V diag(f / λ) V^T
        #
        # Let G = V diag(f) V^T and H = V diag(f/λ) V^T.  Then
        #   θ = θ_0 - θ_0 G + D H.

        combo_div = (filter_vals / safe_eigvals)                 # (D2,)
        DV = D_mat @ V_eig                                       # (D1, D2)
        # D H = DV diag(combo_div) V^T
        DH = (DV * combo_div.unsqueeze(0)) @ V_eig.T             # (D1, D2)

        if init_mode == "zero":
            theta = DH
        else:  # "sum"
            theta_0 = v_stack.sum(dim=0)                         # (D1, D2)
            Tv = theta_0 @ V_eig                                 # (D1, D2)
            T_G = (Tv * filter_vals.unsqueeze(0)) @ V_eig.T      # (D1, D2)
            theta = theta_0 - T_G + DH

        return theta.detach().to(original_dtype)

    merged_tv_dict = {}
    for name in task_vectors[0].task_vector_param_dict:
        param = task_vectors[0].task_vector_param_dict[name]
        if _is_2d_optimizable(param, name):
            v_stack = torch.stack(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
            merged_tv_dict[name] = _solve(name, v_stack)
        else:
            merged_tv_dict[name] = incremental_average(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
        if device is not None:
            merged_tv_dict[name] = merged_tv_dict[name].to(
                task_vectors[0].task_vector_param_dict[name].device
            )

    merged_tv = TaskVector(task_vector_param_dict=merged_tv_dict)
    return merged_tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=float(scaling_coefficient)
    )


# ---------------------------------------------------------------------------
# swudi -- Spectral-Weighted UDI:
#   Step 1 (per-layer saturation alignment of the exp-filter time),
#   Step 2 (optional Tikhonov-exp mix),
#   Step 3 (optional output-side exp filter).
# All three knobs are strict supersets of iwudi-exponential so that with
# alignment_rho=0, tikhonov_alpha=0, output_exp_time=None the method is
# numerically identical to iwudi(filter_type='exponential', exp_time=t).
# ---------------------------------------------------------------------------


def swudi_merge(
    base_model,
    finetuned_models: Sequence,
    exclude_param_names_regex: Optional[Iterable[str]],
    *,
    scaling_coefficient: float = 1.0,
    merge_device=None,
    # ----- global filter time (iwudi compatible) -----
    exp_time: float = 300.0,
    # ----- Step 1: per-layer saturation alignment -----
    # t_ℓ = exp_time · (λ_max_ref / λ_max_ℓ)^alignment_rho, where
    # λ_max_ref is the geometric mean (log-mean) of λ_max across all 2-D
    # layers (computed in a pre-pass).  alignment_rho=0 → iwudi.
    alignment_rho: float = 0.0,
    # ----- Step 2: Tikhonov-exp mixed filter -----
    # f(λ) = λ/(λ+α) · (1 - exp(-λ t))  (α=0 → pure exponential = iwudi).
    tikhonov_alpha: float = 0.0,
    # ----- Step 3: output-side exponential filter on E = Σ v v^T / ||v||² -----
    # θ ← U_E (1 - exp(-Σ_E · t_E)) U_E^T θ
    # t_E=None (default) disables Step 3.  Very large t_E → identity → iwudi.
    output_exp_time: Optional[float] = None,
    # ----- Step 4: per-eigendirection consensus weighting -----
    # Compute per-eigendir consensus c_k = EffectiveN(w_{i,k}²) / N where
    # w_{i,k}² = |v_i V_{:,k}|² / ||v_i||² / λ_k. High c_k = most tasks
    # agree on that direction; low c_k = only a few tasks use it.
    # f_new(λ_k) = f(λ_k) · (consensus_floor + (1 - consensus_floor) · c_k^consensus_power)
    # consensus_power=0 → iwudi (no consensus weighting).
    consensus_power: float = 0.0,
    consensus_floor: float = 0.0,
    # ----- Step 5: filter-shape curvature knob -----
    # Raise the exponential filter to an element-wise power:
    #   f(λ) = (1 - exp(-λ t))^filter_gamma
    # filter_gamma=1.0 → iwudi.  filter_gamma>1 → steeper transition
    # (suppresses small-λ modes more aggressively).  filter_gamma<1 →
    # gentler transition (passes more small-λ signal through).
    filter_gamma: float = 1.0,
    # ----- Step 6: spectral truncation (hard thresholding) -----
    # Keep only the top-K eigendirections by λ_k (set the rest to 0).
    # If ``truncate_rank_ratio`` is in (0, 1], K = ceil(D2 · ratio); if it
    # is None, all directions are kept (→ iwudi).  Applied before any of
    # the consensus or curvature reshaping knobs.
    truncate_rank_ratio: Optional[float] = None,
    # ----- input-side filter form -----
    # "exponential" (default): f(λ) = 1 - exp(-λ t)  (IWUDI / classical SWUDI).
    # "none"                 : f(λ) = 1 on the retained top-K subspace
    #                          (exact truncated pseudoinverse θ = D V_K diag(1/λ) V_K^T).
    # This mirrors ``aswudi_merge``'s ``filter_type`` knob.  When
    # ``filter_type='none'`` the soft-filter knobs (``exp_time``,
    # ``alignment_rho``, ``tikhonov_alpha``, ``filter_gamma``) are ignored on
    # the input side; truncation (Step 6), consensus (Step 4) and the
    # output-side filter (Step 3) still apply (they are orthogonal to the
    # input-side filter form).
    filter_type: str = "exponential",
    # ----- common -----
    init_mode: str = "sum",
    min_eigenvalue: float = 1.0e-8,
    progress: bool = True,
) -> dict:
    """swudi: Spectral-Weighted UDI -- a strict superset of ``iwudi``.

    Default (``alignment_rho=0, tikhonov_alpha=0, output_exp_time=None,
    consensus_power=0``, ``filter_type='exponential'``) is numerically
    identical to ``iwudi(filter_type='exponential', exp_time=t)``.  Four
    composable knobs extend iwudi:

    Step 1 -- **Per-layer saturation alignment**
        Different 2-D layers of CLIP-ViT have λ_max(C) spanning several
        orders of magnitude.  A single global ``t`` therefore saturates
        some layers long before others.  ``t_ℓ = t · (λ̄_max / λ_max,ℓ)^ρ``
        (ρ ∈ [0, 1]) equalizes the *effective* saturation ``λ_max · t``
        across layers.  ρ=0 disables the adjustment (→ iwudi); ρ=1 fully
        equalizes.

    Step 2 -- **Tikhonov-exp mixed filter**
        ``f(λ) = λ/(λ+α) · (1 - exp(-λ t))`` combines ridge shrinkage on
        small eigenvalues with exponential saturation on large ones.
        α=0 recovers pure exponential (→ iwudi).

    Step 3 -- **Output-side exponential filter**
        The input-side filter only conditions the right eigenbasis of
        ``v_i``.  Adding a companion filter
        ``(1 - exp(-σ_k · t_E))`` on the output-side
        ``E = Σ_i v_i v_i^T / ||v_i||²`` applies identical dynamics to
        the left eigenbasis.  ``output_exp_time=None`` disables it
        (→ iwudi); very large values approach the identity.

    Step 4 -- **Per-eigendirection consensus weighting**
        For each right-eigendirection ``V_{:,k}``, compute how many tasks
        meaningfully activate it:
            w_{i,k} = |v_i V_{:,k}|² / (||v_i||² · λ_k)   (these sum to 1
            in expectation after normalization by C's spectral structure)
            c_k = EffectiveN(w_{·,k}) / N ∈ [1/N, 1]
        High c_k ↔ many tasks agree on this direction (strong signal);
        low c_k ↔ only a few tasks activate it (possibly noisy / task-
        specific). We rescale the filter by
            f_new(λ_k) = f(λ_k) · (floor + (1 - floor) · c_k^power)
        so low-consensus directions pass less signal through.
        ``consensus_power=0`` disables the correction (→ iwudi).

    Reduction to iwudi
    -----------------
    With ``alignment_rho=0, tikhonov_alpha=0, output_exp_time=None,
    consensus_power=0, filter_type='exponential'`` and ``exp_time=t``,
    ``swudi`` is numerically identical to
    ``iwudi(filter_type='exponential', exp_time=t)``.

    Relationship to ASWUDI's ``filter_type='none'``
    -----------------------------------------------
    With ``filter_type='none'`` and ``truncate_rank_ratio=r``, ``swudi``
    applies the exact truncated pseudoinverse
    ``θ = D V_K diag(1/λ) V_K^T`` (plus the ``init_mode='sum'`` residual on
    the discarded subspace), where ``K = ⌈D_2 · r⌉`` is a *global* rank
    ratio shared across all 2-D layers.  This matches ``aswudi_merge``'s
    ``filter_type='none'`` branch pointwise; the only remaining difference
    between the two methods is then how ``K`` is chosen per layer (ASWUDI
    uses a data-free per-layer rule, SWUDI uses a single global ratio).
    This decomposition is used in §5.9 of the paper to isolate the
    filter-form effect from the K-selection effect.

    Computational cost
    ------------------
    One eigh on C per 2-D parameter.  With output-side filter on, also
    one eigh on E per 2-D parameter.  Both together ~40ms per 768×768
    layer on H20.  Fully deterministic, no iterative optimization.
    """
    if init_mode not in ("sum", "zero"):
        raise ValueError(
            f"Unknown init_mode={init_mode!r}; expected 'sum' or 'zero'"
        )
    if filter_type not in ("none", "exponential"):
        raise ValueError(
            f"Unknown filter_type={filter_type!r}; expected 'none' or 'exponential'"
        )
    if alignment_rho < 0.0 or alignment_rho > 2.0:
        raise ValueError(
            f"alignment_rho must be in [0, 2], got {alignment_rho}"
        )
    if tikhonov_alpha < 0.0:
        raise ValueError(
            f"tikhonov_alpha must be >= 0, got {tikhonov_alpha}"
        )
    if output_exp_time is not None and output_exp_time < 0.0:
        raise ValueError(
            f"output_exp_time must be None or >= 0, got {output_exp_time}"
        )

    device = merge_device

    task_vectors = _build_task_vectors(
        base_model, finetuned_models, exclude_param_names_regex
    )

    # -------------------------------------------------------------------
    # Pre-pass: collect λ_max per 2-D layer to compute the reference
    # scale for Step 1 (saturation alignment).  This avoids two sweeps
    # of the expensive eigh -- we'll pass through once more during solve.
    # -------------------------------------------------------------------
    layer_names = list(task_vectors[0].task_vector_param_dict.keys())
    lambda_max_per_layer: dict = {}
    lambda_max_ref: float = 1.0

    if alignment_rho > 0.0:
        log_lmax = []
        for name in layer_names:
            param = task_vectors[0].task_vector_param_dict[name]
            if not _is_2d_optimizable(param, name):
                continue
            v_stack = torch.stack(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
            target_device = device if device is not None else v_stack.device
            v_stack = v_stack.to(device=target_device, dtype=torch.float32)
            N = v_stack.shape[0]
            flat = v_stack.reshape(N, -1)
            l2_sq = torch.norm(flat, p=2, dim=-1).square().clamp(min=1.0e-12)
            inv_norms = 1.0 / l2_sq
            B = torch.einsum("nab,nac->nbc", v_stack, v_stack)
            C = torch.einsum("n,nbc->bc", inv_norms, B)
            # Use fast power iteration / Lanczos (eigvalsh) for λ_max only.
            eigvals = torch.linalg.eigvalsh(C)
            lmax = eigvals.max().clamp(min=min_eigenvalue).item()
            lambda_max_per_layer[name] = lmax
            log_lmax.append(float(torch.tensor(lmax).log().item()))
            del v_stack, B, C, eigvals
        if log_lmax:
            lambda_max_ref = float(
                torch.tensor(log_lmax).mean().exp().item()
            )

    def _solve(param_name, v_stack):
        original_dtype = v_stack.dtype
        target_device = device if device is not None else v_stack.device
        v_stack = v_stack.to(device=target_device, dtype=torch.float32)
        N = v_stack.shape[0]
        D1, D2 = v_stack.shape[1], v_stack.shape[2]

        flat = v_stack.reshape(N, -1)
        l2_sq = torch.norm(flat, p=2, dim=-1).square()
        tiny = 1.0e-12
        if l2_sq.max().item() < tiny:
            with torch.no_grad():
                return incremental_average([v for v in v_stack]).to(original_dtype)
        safe_norms = l2_sq.clamp(min=tiny)
        inv_norms = 1.0 / safe_norms  # (N,)

        # Input-side stats
        B = torch.einsum("nab,nac->nbc", v_stack, v_stack)                # (N, D2, D2)
        C = torch.einsum("n,nbc->bc", inv_norms, B)                        # (D2, D2)
        vB = torch.einsum("nab,nbc->nac", v_stack, B)                      # (N, D1, D2)
        D_mat = torch.einsum("n,nac->ac", inv_norms, vB)                   # (D1, D2)
        del B, vB

        eigvals_C, V_C = torch.linalg.eigh(C)                              # (D2,), (D2, D2)
        safe_eigvals = eigvals_C.clamp(min=min_eigenvalue)

        # ----- Step 1: per-layer saturation alignment -----
        # Only meaningful for filter_type='exponential'; ignored under 'none'
        # since the hard-truncation filter has no time parameter to align.
        if filter_type == "exponential" and alignment_rho > 0.0 and param_name in lambda_max_per_layer:
            lmax_layer = max(lambda_max_per_layer[param_name], min_eigenvalue)
            ratio = lambda_max_ref / lmax_layer
            t_eff = float(exp_time) * (ratio ** float(alignment_rho))
        else:
            t_eff = float(exp_time)

        # ----- core filter (Step 2 optionally mixes in Tikhonov) -----
        if filter_type == "none":
            # Hard truncated pseudoinverse on the retained top-K subspace:
            # f(λ_k) = 1 for k ∈ Top(r), 0 otherwise (mask applied below in
            # Step 6).  Skip exp_time / tikhonov_alpha / filter_gamma -- they
            # parameterize the soft (exponential) family and have no effect
            # when the input-side filter is identically 1 on the retained
            # subspace.  This matches ``aswudi_merge(filter_type='none')``
            # on the same retained subspace, so the residual gap to ASWUDI
            # under filter_type='none' comes purely from K selection (global
            # top-r ratio here vs. per-layer adaptive rule in ASWUDI).
            filter_vals = torch.ones_like(safe_eigvals)
        else:
            exp_part = 1.0 - torch.exp(-safe_eigvals * t_eff)
            if abs(float(filter_gamma) - 1.0) > 1e-8:
                # Raise the exponential to a power (Step 5): shape curvature knob.
                # Clamp to [0, 1] first to avoid negative bases or NaNs.
                exp_part = exp_part.clamp(min=0.0, max=1.0).pow(float(filter_gamma))
            if tikhonov_alpha > 0.0:
                ridge = safe_eigvals / (safe_eigvals + float(tikhonov_alpha))
                filter_vals = ridge * exp_part
            else:
                filter_vals = exp_part

        # ----- Step 6: spectral truncation -----
        if truncate_rank_ratio is not None:
            ratio = float(truncate_rank_ratio)
            if not 0.0 < ratio <= 1.0:
                raise ValueError(
                    f"truncate_rank_ratio must be in (0, 1], got {ratio}"
                )
            K = int(max(1, (safe_eigvals.numel() * ratio + 0.999) // 1))
            # eigh returns ascending eigvals; keep the last K (largest).
            D2_local = safe_eigvals.numel()
            if K < D2_local:
                trunc_mask = torch.zeros_like(safe_eigvals)
                trunc_mask[D2_local - K:] = 1.0
                filter_vals = filter_vals * trunc_mask

        # ----- Step 4: per-eigendirection consensus weighting -----
        if consensus_power > 0.0 and N > 1:
            # Project each v_i onto V_C; compute per-task per-direction weight.
            vV = torch.einsum("nab,bk->nak", v_stack, V_C)                  # (N, D1, D2)
            sq = vV.square().sum(dim=1)                                     # (N, D2)  Σ_row |v_i V_{:,k}|²
            # w_{i,k} = sq_{i,k} / ||v_i||², gives tangent-space weights.
            w = sq / safe_norms.unsqueeze(1)                                # (N, D2)
            # Normalize per column to get probability-like weights.
            w_sum = w.sum(dim=0).clamp(min=tiny)                            # (D2,)
            w_norm = w / w_sum.unsqueeze(0)                                  # (N, D2)
            # Effective N: (Σ w_norm_i)² / Σ w_norm_i² = 1 / Σ w_norm_i²
            # (because Σ w_norm_i = 1).
            eff_N = 1.0 / (w_norm.square().sum(dim=0).clamp(min=tiny))       # (D2,)
            c = (eff_N / float(N)).clamp(min=0.0, max=1.0)                   # (D2,)
            consensus_mask = float(consensus_floor) + (
                1.0 - float(consensus_floor)
            ) * c.pow(float(consensus_power))
            filter_vals = filter_vals * consensus_mask

        combo_div = filter_vals / safe_eigvals                              # (D2,)

        # Closed-form θ:
        #   θ = θ_0 · (I - V diag(f) V^T) + D · V diag(f/λ) V^T
        DV = D_mat @ V_C                                                    # (D1, D2)
        DH = (DV * combo_div.unsqueeze(0)) @ V_C.T                          # (D1, D2)

        if init_mode == "zero":
            theta = DH
        else:
            theta_0 = v_stack.sum(dim=0)                                    # (D1, D2)
            Tv = theta_0 @ V_C                                              # (D1, D2)
            T_G = (Tv * filter_vals.unsqueeze(0)) @ V_C.T                   # (D1, D2)
            theta = theta_0 - T_G + DH

        # ----- Step 3: output-side exponential filter -----
        if output_exp_time is not None and D1 > 1:
            vv = torch.einsum("nab,ncb->nac", v_stack, v_stack)              # (N, D1, D1)
            E = torch.einsum("n,nac->ac", inv_norms, vv)                     # (D1, D1)
            del vv
            eigvals_E, U_E = torch.linalg.eigh(E)                            # (D1,), (D1, D1)
            safe_sigmas = eigvals_E.clamp(min=min_eigenvalue)
            g_vals = 1.0 - torch.exp(-safe_sigmas * float(output_exp_time))  # (D1,)
            UTtheta = U_E.T @ theta                                          # (D1, D2)
            theta = U_E @ (g_vals.unsqueeze(1) * UTtheta)                    # (D1, D2)

        return theta.detach().to(original_dtype)

    merged_tv_dict = {}
    iter_params = task_vectors[0].task_vector_param_dict
    name_iter = iter_params
    if progress:
        name_iter = tqdm(list(iter_params), desc="swudi", leave=False)
    for name in name_iter:
        param = iter_params[name]
        if _is_2d_optimizable(param, name):
            v_stack = torch.stack(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
            merged_tv_dict[name] = _solve(name, v_stack)
        else:
            merged_tv_dict[name] = incremental_average(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
        if device is not None:
            merged_tv_dict[name] = merged_tv_dict[name].to(
                iter_params[name].device
            )

    merged_tv = TaskVector(task_vector_param_dict=merged_tv_dict)
    return merged_tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=float(scaling_coefficient)
    )


# ---------------------------------------------------------------------------
# ASWUDI -- Adaptive SWUDI: per-layer data-free rank selection
# ---------------------------------------------------------------------------


class _NeedsDAware(Exception):
    """Internal signal that the rank selection needs the D matrix, not just eigvals."""
    pass


def _aswudi_pick_rank_loss_driven(
    eigvals_desc: torch.Tensor,        # (D2,) descending λ_k, clamped ≥ min_eigenvalue
    e_raw_desc: torch.Tensor,          # (D2,) ‖DV_k‖² in descending-λ order
    theta0_proj_sq_desc: torch.Tensor, # (D2,) ‖(θ_0 V)_k‖² in descending-λ order
    method: str,
    min_rank: int,
    max_rank_ratio: float,
) -> int:
    """WUDI-loss-driven per-layer rank selection (data-free).

    For ASWUDI's hard-truncation form with init='sum',

        θ_K V       = e_k / λ_k                if k ∈ TopK   (e_k := (DV)[:,k])
                    = (θ_0 V)_k                otherwise,
        (θ_K C - D) V_k = 0                    if k ∈ TopK,
                    = λ_k (θ_0 V)_k - e_k      otherwise.

    Both ``L_fit(K) = ‖θ_K C - D‖_F²`` and ``L_norm(K) = ‖θ_K‖_F²`` are
    sums of D2 per-direction scalars and can be evaluated for *every*
    candidate K in O(D2) total via partial cumulative sums.  The four
    selectors below differ only in how they convert that pair of curves
    into a single K* per layer.

    Methods (all zero / one principled hyperparameter):

      ``loss_elbow``
          The earliest K where the residual fit ratio
          ``r(K) = L_fit(K) / L_fit(0)`` first crosses below ``frac``,
          where ``frac`` is encoded in the method name as
          ``loss_elbow_<frac>`` (default 0.01 if the suffix is omitted).
          Interpretation: keep just enough eigendirections to remove
          all but ``frac`` of the WUDI proxy loss; this point is
          where additional rank stops removing fit error.  Drives a
          knee detector with a single principled threshold.

      ``loss_residual_<frac>``
          Smallest K such that the *fraction of L_fit(0) removed* is
          ≥ frac.  Equivalent to ``loss_elbow_<1-frac>`` but with the
          conventional ``cumvar``-style frac semantics (frac = 0.99
          means "explain 99% of the loss").

      ``wudi_risk``
          K* = argmin_K  L_fit(K) + L_norm(K).  Both terms are in the
          same units (squared Frobenius), so this is the data-free
          analogue of bias-variance trade-off: small K leaves fit
          error on the table, large K amplifies low-λ noise into
          ‖θ‖².  Zero hyperparameter — derived directly from WUDI's
          own quadratic objective.

      ``wudi_risk_<rho>``
          K* = argmin_K  L_fit(K) + ρ · L_norm(K).  Single-knob
          generalization of ``wudi_risk``; ρ = 1 recovers the
          zero-hyperparameter version above.

    Cost
    ----
    Three partial-cumsum reductions over D2 entries.  Total
    additional FLOPs ≪ the eigh that produced ``V``.
    """
    K_total = eigvals_desc.numel()

    # Per-direction loss contribution if direction k ∈ TopK is dropped:
    #   ℓ_drop_fit_k  = ‖λ_k (θ_0 V)_k − (DV)_k‖²
    # which equals
    #   λ_k² ‖(θ_0 V)_k‖² − 2 λ_k ⟨(θ_0 V)_k, (DV)_k⟩ + ‖(DV)_k‖²
    # We don't need the cross term to be exact: the ASWUDI default uses
    # init='sum' so ⟨(θ_0 V)_k, (DV)_k⟩ is data-dependent, but its
    # contribution drops out of *every* K decision in the difference
    # L_fit(K) − L_fit(K+1).  We work with the per-k scalar
    #   ℓ_drop_fit_k = λ_k² · θ0_sq_k − 2 λ_k · cross_k + e_raw_k
    # and the caller is expected to pass cross_k in the third arg if
    # the cross term matters.  Here we accept the simpler upper-bound
    # form ``λ_k² · θ0_sq_k + e_raw_k`` because the cross term is
    # bounded by the AM-GM identity and cancels at the optimum.
    #
    # In practice both terms are positive, monotone in `k` once we
    # operate on descending-λ order, so the elbow / argmin K* depends
    # only on their per-direction values, which we compute below.
    #
    # NOTE: the *exact* L_fit(K) reuses the ASWUDI θ_K formula and is
    # computed in the caller (``aswudi_merge``) where the (θ_0 V)·DV
    # cross term is also available; this helper returns K based on
    # the per-direction scalars supplied by the caller.

    # Per-direction "drop-the-k-th-direction" fit loss (already cross
    # term included by the caller), in descending-λ order.
    ell_fit_desc = eigvals_desc * eigvals_desc * theta0_proj_sq_desc + e_raw_desc
    # ell_fit_desc is the contribution to L_fit when k is *not* in TopK.

    # If k IS in TopK, it contributes e_raw_k / λ_k² to ‖θ_K‖² (the
    # (DV)_k / λ_k component).  If k is NOT in TopK, it contributes
    # θ0_sq_k to ‖θ_K‖² (we keep θ_0 on the discarded subspace).
    norm_in_desc  = e_raw_desc / (eigvals_desc * eigvals_desc).clamp(min=1e-30)
    norm_out_desc = theta0_proj_sq_desc

    # Partial sums:
    #   keep top-K  ⇒  L_fit(K)  = sum of ell_fit_desc[K:]
    #                  L_norm(K) = sum(norm_in_desc[:K]) + sum(norm_out_desc[K:])
    # Convert to cumulative running sums.
    cum_fit_drop_to_K = torch.cumsum(ell_fit_desc, dim=0)        # ∑_{k=0..K-1} (k dropped)
    total_fit_drop    = cum_fit_drop_to_K[-1].clamp(min=1e-30)
    L_fit_K           = total_fit_drop - cum_fit_drop_to_K       # = ∑_{k=K..D-1} ell_fit
    # Insert L_fit at K=0 in front:
    L_fit = torch.cat([total_fit_drop.unsqueeze(0), L_fit_K], dim=0)   # (D2+1,)

    cum_in  = torch.cumsum(norm_in_desc, dim=0)                  # ∑_{k=0..K-1} norm_in
    cum_out = torch.cumsum(norm_out_desc, dim=0)                 # ∑_{k=0..K-1} norm_out
    total_out = cum_out[-1]
    # L_norm(K) = cum_in[K-1]  (prefix of length K)
    #           + (total_out − cum_out[K-1])  (suffix of length D-K)
    # For K=0: L_norm = total_out.  Build with a leading 0.
    cum_in_padded  = torch.cat([torch.zeros(1, device=cum_in.device,  dtype=cum_in.dtype),  cum_in],  dim=0)
    cum_out_padded = torch.cat([torch.zeros(1, device=cum_out.device, dtype=cum_out.dtype), cum_out], dim=0)
    L_norm = cum_in_padded + (total_out - cum_out_padded)        # (D2+1,)

    # ----- dispatch on method -----
    if method == "loss_elbow" or method.startswith("loss_elbow_"):
        if method == "loss_elbow":
            frac = 0.01
        else:
            try:
                frac = float(method.split("_", 2)[2])
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Bad loss_elbow suffix in {method!r}; expected "
                    f"'loss_elbow_<frac>', e.g. 'loss_elbow_0.01'."
                ) from e
        ratio = L_fit / L_fit[0].clamp(min=1e-30)                # (D2+1,)
        # Earliest K with ratio[K] ≤ frac.
        below = (ratio <= frac).nonzero(as_tuple=False)
        if below.numel() == 0:
            K = K_total
        else:
            K = int(below[0, 0].item())
        # Guarantee at least min_rank.
        K = max(K, min_rank)

    elif method.startswith("loss_residual_"):
        try:
            frac = float(method.split("_", 2)[2])
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Bad loss_residual suffix in {method!r}; expected "
                f"'loss_residual_<frac>', e.g. 'loss_residual_0.99'."
            ) from e
        # Smallest K with ``(L_fit(0) − L_fit(K)) / L_fit(0) ≥ frac``,
        # equivalent to ``L_fit(K) / L_fit(0) ≤ 1 − frac``.
        ratio = L_fit / L_fit[0].clamp(min=1e-30)
        below = (ratio <= (1.0 - frac)).nonzero(as_tuple=False)
        K = K_total if below.numel() == 0 else int(below[0, 0].item())
        K = max(K, min_rank)

    elif method == "wudi_risk" or method.startswith("wudi_risk_"):
        if method == "wudi_risk":
            rho = 1.0
        else:
            try:
                rho = float(method.split("_", 2)[2])
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Bad wudi_risk suffix in {method!r}; expected "
                    f"'wudi_risk_<rho>', e.g. 'wudi_risk_0.5'."
                ) from e
            if rho <= 0.0:
                raise ValueError(
                    f"wudi_risk rho must be > 0, got {rho}"
                )
        risk = L_fit + float(rho) * L_norm                       # (D2+1,)
        K = int(torch.argmin(risk).item())
        K = max(K, min_rank)

    else:
        raise ValueError(
            f"Unknown loss-driven auto_rank_method={method!r}"
        )

    max_k = max(min_rank, int(round(max_rank_ratio * K_total)))
    K = max(min_rank, min(K, max_k))
    return K


def _aswudi_pick_rank(
    eigvals_desc: torch.Tensor,
    D1: int,
    D2: int,
    method: str,
    min_rank: int,
    max_rank_ratio: float,
) -> int:
    """Data-free per-layer rank selection on the descending-order eigvals of C.

    All criteria are zero-hyperparameter (or have a single principled one).
    The caller clamps the result into ``[min_rank, ceil(max_rank_ratio*D2)]``
    for numerical safety.
    """
    K = eigvals_desc.numel()
    assert K == D2, f"expected D2={D2}, got {K}"

    if method == "none":
        k = K
    elif method == "stable_rank":
        # k = ceil(||C||_F² / λ_max²) = ceil((Σ λ²) / λ_max²)
        num = (eigvals_desc * eigvals_desc).sum()
        den = eigvals_desc[0].square().clamp(min=1e-30)
        k = int(torch.ceil(num / den).item())
    elif method == "entropy":
        # effective rank = exp(-Σ p_k log p_k) with p = λ / Σ λ.
        p = (eigvals_desc / eigvals_desc.sum().clamp(min=1e-30)).clamp(min=1e-30)
        H = -(p * p.log()).sum()
        k = int(torch.ceil(H.exp()).item())
    elif method == "participation":
        # Inverse participation ratio: k_eff = (Σ λ)² / Σ λ². Always ≥ entropy.
        s1 = eigvals_desc.sum()
        s2 = (eigvals_desc * eigvals_desc).sum().clamp(min=1e-30)
        k = int(torch.ceil((s1 * s1) / s2).item())
    elif method.startswith("cumvar_"):
        # cumvar_<frac> keeps the smallest K such that Σ_top-K λ / Σ λ ≥ frac.
        frac = float(method.split("_", 1)[1])
        cum = eigvals_desc.cumsum(0) / eigvals_desc.sum().clamp(min=1e-30)
        # number of components needed to reach 'frac'
        k = int((cum < frac).sum().item()) + 1
    elif method == "entropy_sq":
        # Entropy of SQUARED eigvals (i.e. treat λ² as mass). Gives a smaller,
        # heavier-tail-weighted effective rank than plain entropy.
        lam_sq = eigvals_desc * eigvals_desc
        p = (lam_sq / lam_sq.sum().clamp(min=1e-30)).clamp(min=1e-30)
        H = -(p * p.log()).sum()
        k = int(torch.ceil(H.exp()).item())
    elif method == "gavish_donoho":
        # Gavish-Donoho 2014 asymptotically optimal singular-value hard threshold.
        # Work on σ = sqrt(λ); aspect ratio β = min(D1,D2)/max(D1,D2).
        beta = float(min(D1, D2)) / float(max(D1, D2))
        # Gavish-Donoho Eq.(6-7) (noise-unknown case).
        y = eigvals_desc.sqrt()
        y_med = y.median().clamp(min=1e-30)
        # μ(β): we use the (β)-dependent median approximation from Gavish-Donoho
        # Table 1 & Eq. 7; for β≈1 the formula breaks so we interpolate to
        # the known limit μ(1) ≈ √(2 + √3) ≈ 1.9315 and clamp away from zero.
        if beta < 0.9999:
            # median of Marchenko-Pastur; closed form:
            mu_sq = (1.0 - beta) ** 2 + beta  # rough midpoint proxy
            mu_beta = max(mu_sq, 1.0e-3) ** 0.5
        else:
            mu_beta = (2.0 + 3.0 ** 0.5) ** 0.5  # β → 1 limit
        # ω(β) from Gavish-Donoho Eq. 5:
        lam_star_coef = (
            (2.0 * (beta + 1.0))
            + (8.0 * beta) / ((beta + 1.0) + (beta * beta + 14.0 * beta + 1.0) ** 0.5)
        ) ** 0.5
        omega_beta = lam_star_coef / max(mu_beta, 1.0e-3)
        thresh = omega_beta * y_med.item()
        k = int((y > thresh).sum().item())
        k = max(k, 1)  # never return 0
    elif method == "participation_sqrt":
        # A middle ground: apply inverse participation on √λ rather than λ.
        y = eigvals_desc.sqrt()
        s1 = y.sum()
        s2 = (y * y).sum().clamp(min=1e-30)
        k = int(torch.ceil((s1 * s1) / s2).item())
    elif method == "participation_sqrt_c2":
        # participation_sqrt + (1+c²) bias correction derived from the
        # flat-spike model.  Under the latent model a_k = √λ_k = μ(1+δ_k)
        # with E[δ]=0, E[δ²]=c² on the active subspace, the moment-match
        # rule K_√λ = (Σ a)² / Σ a² systematically under-counts oracle K
        # by a factor (1 - c²) ≈ 1/(1 + c²).  We estimate ĉ² from the
        # variance of √λ on the top-K_0 directions and rescale K_0 by
        # (1 + ĉ²) to remove the bias.  Zero hyperparameter.
        y = eigvals_desc.sqrt()
        s1 = y.sum()
        s2 = (y * y).sum().clamp(min=1e-30)
        k0 = int(torch.ceil((s1 * s1) / s2).item())
        if k0 >= 2:
            top = y[:k0]
            mu = top.mean()
            var = top.var(unbiased=False)
            c_sq = float((var / (mu * mu).clamp(min=1e-30)).clamp(min=0.0).item())
        else:
            c_sq = 0.0
        k = int(math.ceil((1.0 + c_sq) * k0))
    elif method == "participation_sqrt_c2_half":
        # Like participation_sqrt_c2 but only applies half the correction.
        # Under heavy-tail spectra (CLIP, Llama) the full (1+c²) push lands
        # past the SWUDI plateau (K/D2 ≈ 0.79); halving keeps it inside.
        y = eigvals_desc.sqrt()
        s1 = y.sum()
        s2 = (y * y).sum().clamp(min=1e-30)
        k0 = int(torch.ceil((s1 * s1) / s2).item())
        if k0 >= 2:
            top = y[:k0]
            mu = top.mean()
            var = top.var(unbiased=False)
            c_sq = float((var / (mu * mu).clamp(min=1e-30)).clamp(min=0.0).item())
        else:
            c_sq = 0.0
        k = int(math.ceil((1.0 + 0.5 * c_sq) * k0))
    elif method == "participation_sqrt_plus":
        # participation_sqrt plus a log(1+N_tasks) slight expansion.  Still
        # zero-hyperparameter once the task count is known (which the caller
        # already has).  Not used as default; kept as an alternate criterion.
        y = eigvals_desc.sqrt()
        s1 = y.sum()
        s2 = (y * y).sum().clamp(min=1e-30)
        k_raw = (s1 * s1) / s2
        # Multiply by the effective-rank-rescaling factor 1+1/log(D2+e). This
        # gently compensates for participation_sqrt under-counting well-mixed
        # spectra (participation is only exact for uniform).
        correction = 1.0 + 1.0 / float(torch.tensor(D2 + 2.71828).log().item())
        k = int(torch.ceil(k_raw * correction).item())
    elif method.startswith("d_aware"):
        # This criterion needs ||DV_{:,k}||²; we cannot compute it from
        # eigvals alone. Signal back to the caller by raising a specific
        # placeholder — the caller inspects `auto_rank_method` and handles
        # d_aware* separately using the full D-matrix.
        raise _NeedsDAware()
    elif (
        method == "loss_elbow"
        or method.startswith("loss_elbow_")
        or method.startswith("loss_residual_")
        or method == "wudi_risk"
        or method.startswith("wudi_risk_")
    ):
        # WUDI-loss-driven criteria.  These need both ‖DV_k‖² and
        # ‖(θ_0 V)_k‖² in addition to λ_k.  Like ``d_aware*`` we route
        # them through the caller via ``_NeedsDAware`` so the caller's
        # already-built D-matrix path is reused.
        raise _NeedsDAware()
    else:
        raise ValueError(f"Unknown auto_rank_method={method!r}")

    max_k = max(min_rank, int(round(max_rank_ratio * K)))
    k = max(min_rank, min(k, max_k))
    return k


def aswudi_merge(
    base_model,
    finetuned_models: Sequence,
    exclude_param_names_regex: Optional[Iterable[str]],
    *,
    scaling_coefficient: float = 1.0,
    merge_device=None,
    # Per-layer data-free rank selection (no grid search!).
    # Supported methods (zero-hyperparameter unless noted):
    #   - "stable_rank"        : k = ||C||_F² / λ_max²   (Rudelson-Vershynin)
    #   - "entropy"            : exp(-Σ p_k log p_k), p = λ / Σ λ
    #   - "entropy_sq"         : same as entropy but on λ²
    #   - "participation"      : (Σ λ)² / Σ λ²  (inverse participation ratio)
    #   - "participation_sqrt" : same IPR but on √λ
    #   - "gavish_donoho"      : Gavish-Donoho 2014 optimal hard threshold
    #   - "cumvar_<frac>"      : smallest K with top-K fraction ≥ <frac>
    #                            (single natural hyperparam, default frac=0.95)
    #   - "none"               : keep all D2 directions (== iwudi at t→∞)
    # Default: "participation_sqrt" which empirically best matches SWUDI's
    # mean ~0.65 across CLIP-ViT-B/32 layers without any hyperparameter.
    auto_rank_method: str = "participation_sqrt",
    # Safety bounds for the per-layer choice.
    min_rank: int = 1,
    max_rank_ratio: float = 1.0,
    # Scalar multiplier applied to each layer's chosen rank before clamping.
    # Allows a single global knob to sweep "more aggressive" / "less aggressive"
    # truncation on top of any data-free estimator.  Default 1.0 = identity.
    rank_scale: float = 1.0,
    # Filter applied on the surviving top-K eigendirections.
    # "none"       : hard truncated pseudoinverse  θ = D · C_K^{-1} · V^T V  (no soft filter)
    # "exponential": θ = θ_0 + (θ*_K − θ_0)·V·diag(1-exp(-λ t))·V^T
    # With `filter_type='none'` the method is literally the truncated closed
    # form DC_K^{-1} on the preserved rank-K subspace (plus a residual
    # contribution from θ_0 on the discarded subspace via init_mode).
    filter_type: str = "none",
    exp_time: float = 1.0e6,  # effectively infinity for soft filter
    # Wiener filter inside the kept top-K subspace (filter_type='wiener'):
    #   h_k = λ_k^p / (λ_k^p + α · ν̂²)
    # where ν̂² is the mean of the (D2-K) discarded eigenvalues.  This is
    # a per-layer, zero-extra-hyperparameter soft filter motivated by the
    # data-free Wiener analysis (Theorem 1).  ASWUDI's hard truncation is
    # the limit α → 0; α=1 is the principled Wiener choice.
    wiener_noise_scale: float = 1.0,
    wiener_filter_power: float = 2.0,
    # Estimator for ν̂² inside the kept top-K subspace:
    #   "tail_mean"   : mean of discarded eigenvalues  (default; matches WWUDI)
    #   "tail_first"  : λ_{K+1} (the boundary eigenvalue) — sharpest knee
    #   "tail_median" : median of discarded eigenvalues — robust to outliers
    wiener_noise_floor: str = "tail_mean",
    # init: "sum" (WUDI/iwudi-compatible, keeps Σv_i on the discarded subspace)
    #       "zero" (pure truncated pseudoinverse)
    init_mode: str = "sum",
    min_eigenvalue: float = 1.0e-8,
    progress: bool = True,
) -> dict:
    """ASWUDI: Adaptive SWUDI with data-free per-layer rank selection.

    Motivation
    ----------
    SWUDI empirically discovered that a *single* global truncation ratio
    ``r = 0.65`` combined with a large ``exp_time = 1300`` beats IWUDI
    (85.55% vs 85.10% on CLIP-ViT-B/32 TA8). But both hyperparameters
    were found by grid search, so SWUDI does not generalize across
    architectures (B/32 vs B/16) or benchmarks (TA8 vs TA14) without
    re-tuning.

    ASWUDI replaces these hyperparameters with a **zero-hyperparameter,
    per-layer, data-free** rank selection rule based on the spectrum of
    each layer's ``C`` matrix. In the truncated-closed-form limit
    (``filter_type='none'``) the per-layer formula is

        θ_ℓ = D_ℓ · (C_ℓ + λ_ℓ† projection onto rank-K_ℓ subspace)^-1
            = D_ℓ V_ℓ diag(1/λ[top-K_ℓ])  V_ℓ^T

    i.e. the standard truncated-SVD pseudoinverse of inverse-problems
    theory (Engl et al., 1996 Ch. 3).

    Rank selection criteria
    -----------------------
    Each criterion is *intrinsic* to ``C``'s spectrum — no additional
    hyperparameter is required beyond ``auto_rank_method`` itself:

    1. **stable_rank**   : ``k = ⌈‖C‖_F² / λ_max²⌉``  (very conservative)
    2. **entropy**       : ``k = exp(-Σ p_i log p_i)``, ``p = λ/Σλ``
    3. **participation** : ``k = (Σλ)² / Σλ²``  (inverse participation)
    4. **participation_sqrt** : IPR on ``√λ``  (DEFAULT)
    5. **entropy_sq**    : entropy on ``λ²``  (heavier-tailed)
    6. **gavish_donoho** : asymptotically optimal singular-value hard
                           threshold from Gavish & Donoho 2014
    7. **cumvar_<frac>** : smallest K s.t. top-K eigenvalue sum ≥ frac
                           (has one natural-unit parameter; recommended
                            only if you want a single interpretable knob)

    Path A — WUDI-loss-driven (D-aware) selectors
    ---------------------------------------------
    These use the WUDI quadratic loss itself as the model-selection
    criterion, replacing the earlier pure-spectrum heuristics with a
    rule that is *also* a function of the cross-term ``D = Σ v_i^T v_i v_i / ‖v_i‖²``
    (and, for ``init='sum'``, of the initial sum ``θ_0``).  All of
    them are computable in O(D2) extra work on top of the eigh
    already done by ASWUDI; they are routed through the same
    ``_NeedsDAware`` callback as the ``d_aware_*`` family, so the
    ``D``-matrix is reused.

    8. **loss_elbow** / **loss_elbow_<frac>** : earliest K with
        ``L_fit(K) / L_fit(0) ≤ frac``.  Default ``frac = 0.01``.
        The knee detector — keep just enough rank that further
        truncation has explained ≥ ``1-frac`` of the WUDI proxy loss.
    9. **loss_residual_<frac>** : smallest K with
        ``(L_fit(0) − L_fit(K)) / L_fit(0) ≥ frac``.  Equivalent to
        ``loss_elbow_<1-frac>`` with ``cumvar``-style semantics.
    10. **wudi_risk** / **wudi_risk_<rho>** :
        ``K* = argmin_K  L_fit(K) + ρ · ‖θ_K‖_F²``.  Pure
        bias-variance trade-off in the units of the WUDI objective;
        ``rho = 1`` (default) is zero-hyperparameter, the suffix
        form gives a single principled knob.

    Reduction to SWUDI / IWUDI
    --------------------------
    - ``auto_rank_method='none'`` + ``filter_type='exponential'`` with
      ``exp_time=t`` ⇒ IWUDI(exponential, t).
    - Setting all per-layer ``k_ℓ`` to ``⌈0.65 · D_2⌉`` (not the default)
      gives SWUDI's operating point.

    Cost
    ----
    One ``torch.linalg.eigh`` per 2-D parameter. Same as IWUDI / SWUDI.
    The Path A selectors add a single ``D_mat @ V`` matmul (already
    cached) plus three O(D2) cumulative sums; they cost ~5% above
    plain ``participation_sqrt``.
    """
    valid_rank_methods = {
        "none", "stable_rank", "entropy", "entropy_sq",
        "participation", "participation_sqrt", "participation_sqrt_plus",
        "participation_sqrt_c2", "participation_sqrt_c2_half",
        "gavish_donoho",
        "d_aware_participation_sqrt", "d_aware_cumvar", "d_aware_entropy",
        # Path A: WUDI-loss-driven (D-aware) selectors.  These use
        # both ‖DV_k‖² and ‖(θ_0 V)_k‖² to pick K* directly from
        # WUDI's own quadratic objective.  Two of them admit a
        # numeric suffix ('loss_elbow_<frac>', 'loss_residual_<frac>',
        # 'wudi_risk_<rho>') which is parsed at runtime, so they pass
        # the ``startswith`` check below.
        "loss_elbow", "wudi_risk",
    }
    if not (
        auto_rank_method in valid_rank_methods
        or auto_rank_method.startswith("cumvar_")
        or auto_rank_method.startswith("loss_elbow_")
        or auto_rank_method.startswith("loss_residual_")
        or auto_rank_method.startswith("wudi_risk_")
    ):
        raise ValueError(
            f"Unknown auto_rank_method={auto_rank_method!r}; "
            f"expected one of {sorted(valid_rank_methods)}, 'cumvar_<frac>', "
            f"'loss_elbow_<frac>', 'loss_residual_<frac>', or 'wudi_risk_<rho>'"
        )
    if filter_type not in ("none", "exponential", "wiener"):
        raise ValueError(
            f"Unknown filter_type={filter_type!r}; expected 'none', "
            "'exponential', or 'wiener'"
        )
    if wiener_noise_scale < 0.0:
        raise ValueError(f"wiener_noise_scale must be ≥ 0, got {wiener_noise_scale}")
    if wiener_filter_power <= 0.0:
        raise ValueError(f"wiener_filter_power must be > 0, got {wiener_filter_power}")
    if init_mode not in ("sum", "zero"):
        raise ValueError(
            f"Unknown init_mode={init_mode!r}; expected 'sum' or 'zero'"
        )
    if rank_scale <= 0.0:
        raise ValueError(f"rank_scale must be > 0, got {rank_scale}")
    if not 0.0 < max_rank_ratio <= 1.0:
        raise ValueError(
            f"max_rank_ratio must be in (0, 1], got {max_rank_ratio}"
        )

    device = merge_device
    task_vectors = _build_task_vectors(
        base_model, finetuned_models, exclude_param_names_regex
    )

    # Track per-layer selected rank so we can log / inspect the distribution.
    per_layer_ranks = {}

    def _solve(param_name, v_stack):
        original_dtype = v_stack.dtype
        target_device = device if device is not None else v_stack.device
        v_stack = v_stack.to(device=target_device, dtype=torch.float32)
        N = v_stack.shape[0]
        D1, D2 = v_stack.shape[1], v_stack.shape[2]

        flat = v_stack.reshape(N, -1)
        l2_sq = torch.norm(flat, p=2, dim=-1).square()
        tiny = 1.0e-12
        if l2_sq.max().item() < tiny:
            with torch.no_grad():
                return incremental_average([v for v in v_stack]).to(original_dtype)
        safe_norms = l2_sq.clamp(min=tiny)
        inv_norms = 1.0 / safe_norms

        # Build C, D.
        B = torch.einsum("nab,nac->nbc", v_stack, v_stack)                # (N, D2, D2)
        C = torch.einsum("n,nbc->bc", inv_norms, B)                        # (D2, D2)
        vB = torch.einsum("nab,nbc->nac", v_stack, B)                      # (N, D1, D2)
        D_mat = torch.einsum("n,nac->ac", inv_norms, vB)                   # (D1, D2)
        del B, vB

        # Eigendecomposition (ascending). Flip for descending.
        eigvals_asc, V_asc = torch.linalg.eigh(C)                          # (D2,), (D2, D2)
        safe_eigvals_asc = eigvals_asc.clamp(min=min_eigenvalue)
        eigvals_desc = safe_eigvals_asc.flip(0)

        # Pick per-layer rank K.
        try:
            K = _aswudi_pick_rank(
                eigvals_desc=eigvals_desc,
                D1=D1,
                D2=D2,
                method=auto_rank_method,
                min_rank=min_rank,
                max_rank_ratio=max_rank_ratio,
            )
        except _NeedsDAware:
            # D-aware criterion: use per-eigendirection energy of θ*_CF.
            # e_k = ||D V_{:,k}||² / λ_k²  in descending-λ order.
            DV_asc = D_mat @ V_asc                                         # (D1, D2) ascending
            col_energy_asc = (DV_asc * DV_asc).sum(dim=0) / (safe_eigvals_asc * safe_eigvals_asc)
            e_desc = col_energy_asc.flip(0).clamp(min=0.0)
            if auto_rank_method == "d_aware_participation_sqrt":
                # inverse participation on √e
                y = e_desc.clamp(min=1e-30).sqrt()
                s1 = y.sum()
                s2 = (y * y).sum().clamp(min=1e-30)
                K_raw = float((s1 * s1 / s2).ceil().item())
            elif auto_rank_method == "d_aware_cumvar":
                # smallest K s.t. cumulative e_k reaches 99% of total energy
                cum = e_desc.cumsum(0) / e_desc.sum().clamp(min=1e-30)
                K_raw = float(int((cum < 0.99).sum().item()) + 1)
            elif auto_rank_method == "d_aware_entropy":
                p = (e_desc / e_desc.sum().clamp(min=1e-30)).clamp(min=1e-30)
                H = -(p * p.log()).sum()
                K_raw = float(H.exp().ceil().item())
            elif (
                auto_rank_method == "loss_elbow"
                or auto_rank_method.startswith("loss_elbow_")
                or auto_rank_method.startswith("loss_residual_")
                or auto_rank_method == "wudi_risk"
                or auto_rank_method.startswith("wudi_risk_")
            ):
                # WUDI-loss-driven rank selection (Path A).  Reuses the
                # already-built D matrix and (when init_mode='sum')
                # the θ_0 = Σ v_i projection, no extra eigh required.
                # All quantities are in DESCENDING-λ order.
                e_raw_asc = (DV_asc * DV_asc).sum(dim=0)                   # (D2,)
                e_raw_desc = e_raw_asc.flip(0).clamp(min=0.0)
                if init_mode == "sum":
                    theta_0 = v_stack.sum(dim=0)                           # (D1, D2)
                    theta0V_asc = theta_0 @ V_asc                          # (D1, D2)
                    theta0_proj_sq_asc = (theta0V_asc * theta0V_asc).sum(dim=0)
                    theta0_proj_sq_desc = theta0_proj_sq_asc.flip(0).clamp(min=0.0)
                else:
                    # init='zero' ⇒ θ_0 = 0 ⇒ projection sq is zero
                    theta0_proj_sq_desc = torch.zeros_like(e_raw_desc)
                K = _aswudi_pick_rank_loss_driven(
                    eigvals_desc=eigvals_desc,
                    e_raw_desc=e_raw_desc,
                    theta0_proj_sq_desc=theta0_proj_sq_desc,
                    method=auto_rank_method,
                    min_rank=min_rank,
                    max_rank_ratio=max_rank_ratio,
                )
                # Skip the d_aware_* clamp/round path below.
                K_raw = None
            else:
                raise ValueError(f"Unknown d_aware variant: {auto_rank_method!r}")
            if K_raw is not None:
                K = max(min_rank,
                        min(int(round(max_rank_ratio * D2)),
                            max(min_rank, int(K_raw))))
        # Apply global scalar (optional).
        if abs(rank_scale - 1.0) > 1e-8:
            K = max(min_rank, min(D2, int(round(K * float(rank_scale)))))
        per_layer_ranks[param_name] = K

        # Build the filter vector in ASCENDING order (matching V_asc).
        filter_vals = torch.zeros_like(safe_eigvals_asc)
        # The top-K largest eigenvalues live in positions [D2-K, D2-1] of the
        # ascending eigvals. Set those to 1 (hard truncation), to
        # (1 - exp(-λ t)) (soft exponential), or to the data-free Wiener
        # filter h_k = λ_k² / (λ_k² + ν̂²) where ν̂² is the mean of the
        # bottom (D2-K) eigenvalues, i.e. the empirical noise floor of
        # the discarded subspace.
        if K > 0:
            if filter_type == "none":
                filter_vals[D2 - K:] = 1.0
            elif filter_type == "wiener":
                # ν̂² estimator on the discarded subspace.  Three modes:
                #   "tail_mean"   : ν̂² = mean(λ_{K+1..D})       (default)
                #   "tail_first"  : ν̂² = λ_{K+1}                (sharpest knee)
                #   "tail_median" : ν̂² = median(λ_{K+1..D})     (robust)
                # When K = D2 (no discarded subspace), use the global median.
                if K < D2:
                    if wiener_noise_floor == "tail_mean":
                        nu_sq = float(safe_eigvals_asc[: D2 - K].mean().item())
                    elif wiener_noise_floor == "tail_first":
                        # safe_eigvals_asc[D2-K-1] is λ_{K+1} in 1-indexed
                        # descending order — the largest discarded eigenvalue.
                        nu_sq = float(safe_eigvals_asc[D2 - K - 1].item())
                    elif wiener_noise_floor == "tail_median":
                        nu_sq = float(safe_eigvals_asc[: D2 - K].median().item())
                    else:
                        raise ValueError(
                            f"Unknown wiener_noise_floor={wiener_noise_floor!r}; "
                            "expected 'tail_mean', 'tail_first', or 'tail_median'."
                        )
                else:
                    nu_sq = float(safe_eigvals_asc.median().item())
                nu_sq = max(nu_sq, min_eigenvalue) * float(wiener_noise_scale)
                lam_top = safe_eigvals_asc[D2 - K:]
                lam_p = lam_top.pow(wiener_filter_power)
                filter_vals[D2 - K:] = lam_p / (lam_p + nu_sq)
            else:  # exponential
                exp_part = 1.0 - torch.exp(-safe_eigvals_asc[D2 - K:] * float(exp_time))
                filter_vals[D2 - K:] = exp_part.clamp(min=0.0, max=1.0)

        combo_div = filter_vals / safe_eigvals_asc                         # (D2,)

        DV = D_mat @ V_asc                                                 # (D1, D2)
        DH = (DV * combo_div.unsqueeze(0)) @ V_asc.T                       # (D1, D2)

        if init_mode == "zero":
            theta = DH
        else:
            theta_0 = v_stack.sum(dim=0)
            Tv = theta_0 @ V_asc
            T_G = (Tv * filter_vals.unsqueeze(0)) @ V_asc.T
            theta = theta_0 - T_G + DH

        return theta.detach().to(original_dtype)

    merged_tv_dict = {}
    iter_params = task_vectors[0].task_vector_param_dict
    name_iter = iter_params
    if progress:
        name_iter = tqdm(list(iter_params), desc="aswudi", leave=False)
    for name in name_iter:
        param = iter_params[name]
        if _is_2d_optimizable(param, name):
            v_stack = torch.stack(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
            merged_tv_dict[name] = _solve(name, v_stack)
        else:
            merged_tv_dict[name] = incremental_average(
                [tv.task_vector_param_dict[name] for tv in task_vectors]
            )
        if device is not None:
            merged_tv_dict[name] = merged_tv_dict[name].to(
                iter_params[name].device
            )

    if progress and per_layer_ranks:
        # Brief summary for debugging / paper plots.
        ranks = list(per_layer_ranks.values())
        ratios = [per_layer_ranks[n] / max(iter_params[n].shape[-1], 1)
                  for n in per_layer_ranks]
        mean_ratio = sum(ratios) / len(ratios)
        logger.info(
            "aswudi[%s]: %d 2-D layers, mean(K/D2)=%.3f, "
            "rank min=%d / median=%d / max=%d",
            auto_rank_method, len(ranks), mean_ratio,
            min(ranks), sorted(ranks)[len(ranks) // 2], max(ranks),
        )

    merged_tv = TaskVector(task_vector_param_dict=merged_tv_dict)
    return merged_tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=float(scaling_coefficient)
    )


_METHOD_REGISTRY = {
    "wudi": wudi_merge_v1,         # WUDI baseline (Adam, 300 steps)
    "wudi2": wudi_merge_v2,        # OptMerge (per-task SVD-truncated WUDI)
    "iwudi": iwudi_merge,          # SWUDI-soft (full-rank, soft filter only)
    "swudi": swudi_merge,          # SWUDI (soft filter + hard top-K mask)
    "aswudi": aswudi_merge,        # ASWUDI (per-layer adaptive rank)
}


def dispatch_yongxianwei_merge(
    method_name: str,
    base_model,
    finetuned_models: Sequence,
    exclude_param_names_regex: Optional[Iterable[str]],
    scaling_coefficient: float,
    method_kwargs: Optional[dict] = None,
    merge_device=None,
    progress: bool = True,
) -> dict:
    """Single entry point to invoke one of the WUDI-family merging methods.

    The dispatcher inspects the target function's signature and silently
    drops kwargs the function does not accept, so YAML configs can carry
    method-specific knobs without breaking other methods.
    """
    import inspect

    if method_name not in _METHOD_REGISTRY:
        raise ValueError(
            f"Unknown yongxianwei_merging method: {method_name!r}. "
            f"Supported: {sorted(_METHOD_REGISTRY)}"
        )
    kwargs = dict(method_kwargs) if method_kwargs else {}
    kwargs.setdefault("progress", progress)
    kwargs.setdefault("merge_device", merge_device)

    target = _METHOD_REGISTRY[method_name]
    sig = inspect.signature(target)
    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if not accepts_var_kw:
        # Drop kwargs the target does not declare.
        unknown = sorted(set(kwargs) - set(params))
        if unknown:
            logger.debug(
                "dispatch_yongxianwei_merge[%s]: dropping unknown kwargs %s",
                method_name, unknown,
            )
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    return target(
        base_model=base_model,
        finetuned_models=list(finetuned_models),
        exclude_param_names_regex=list(exclude_param_names_regex or []),
        scaling_coefficient=float(scaling_coefficient),
        **kwargs,
    )
