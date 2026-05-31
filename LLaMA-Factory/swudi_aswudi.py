"""Standalone SWUDI / ASWUDI merge implementations for MLLMerging.

Layer-wise spectral-filtering data-free model merging, ported from
fusion_bench/method/yongxianwei_merging/functional.py without any
fusion_bench dependency. Math is identical; we re-use the
TaskVector / get_param_names_to_merge / copy_params_to_model utilities
already defined in model_merging.py.
"""

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


def _is_2d_optimizable(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and "lm_head" not in name


def _incremental_average(tensors: List[torch.Tensor]) -> torch.Tensor:
    out = tensors[0].clone()
    for i, t in enumerate(tensors[1:], 1):
        out += (t - out) / (i + 1)
    return out


def _build_task_vectors(base_model, finetuned_models, exclude_param_names_regex):
    """Returns list[dict[name -> delta]] for each finetuned model."""
    import re
    base_state = {n: p for n, p in base_model.named_parameters()}
    excl = list(exclude_param_names_regex or [])
    keep = [
        n for n in base_state
        if not any(re.match(pat, n) for pat in excl)
    ]
    out = []
    for m in finetuned_models:
        ft_state = {n: p for n, p in m.named_parameters()}
        d = {}
        with torch.no_grad():
            for n in keep:
                d[n] = ft_state[n].detach() - base_state[n].detach()
        out.append(d)
    return out


def _combine_with_pretrained(base_model, merged_tv_dict, scaling_coefficient):
    base_state = {n: p for n, p in base_model.named_parameters()}
    merged = {}
    with torch.no_grad():
        for n, d in merged_tv_dict.items():
            base_p = base_state[n].detach()
            merged[n] = base_p + scaling_coefficient * d.to(
                device=base_p.device, dtype=base_p.dtype
            )
    return merged


# ---------------------------------------------------------------------------
# SWUDI
# ---------------------------------------------------------------------------

def swudi_merge(
    base_model: nn.Module,
    finetuned_models: List[nn.Module],
    exclude_param_names_regex: Iterable[str],
    *,
    scaling_coefficient: float = 1.0,
    merge_device: Optional[str] = "cuda",
    exp_time: float = 300.0,
    truncate_rank_ratio: Optional[float] = 0.65,
    filter_type: str = "exponential",
    init_mode: str = "sum",
    min_eigenvalue: float = 1.0e-8,
    progress: bool = True,
) -> dict:
    """SWUDI: IWUDI exponential-spectral filter + hard top-K truncation.

    Default (exp_time=300, truncate_rank_ratio=0.65) matches the
    CLIP-ViT-B/32 TA8 SOTA configuration from the paper. For LLM/VLM
    full-parameter deltas, prefer (exp_time=300, truncate_rank_ratio=0.85)
    per the Llama tuning result in the paper.

    See ``aswudi_merge`` docstring for the meaning of ``init_mode``; same
    "sum" / "mean" / "zero" semantics apply here.
    """
    assert filter_type in ("exponential", "none"), filter_type
    assert init_mode in ("sum", "mean", "zero"), init_mode

    task_vectors = _build_task_vectors(
        base_model, finetuned_models, exclude_param_names_regex
    )
    device = merge_device

    def _solve(name, v_stack):
        original_dtype = v_stack.dtype
        target = device if device is not None else v_stack.device
        v_stack = v_stack.to(device=target, dtype=torch.float32)
        N, D1, D2 = v_stack.shape
        flat = v_stack.reshape(N, -1)
        l2_sq = torch.norm(flat, p=2, dim=-1).square()
        if l2_sq.max().item() < 1.0e-12:
            return _incremental_average(list(v_stack)).to(original_dtype)
        inv_norms = 1.0 / l2_sq.clamp(min=1.0e-12)

        B = torch.einsum("nab,nac->nbc", v_stack, v_stack)
        C = torch.einsum("n,nbc->bc", inv_norms, B)
        vB = torch.einsum("nab,nbc->nac", v_stack, B)
        D_mat = torch.einsum("n,nac->ac", inv_norms, vB)
        del B, vB

        eigvals, V_C = torch.linalg.eigh(C)
        safe_eigvals = eigvals.clamp(min=min_eigenvalue)

        if filter_type == "none":
            filter_vals = torch.ones_like(safe_eigvals)
        else:
            filter_vals = 1.0 - torch.exp(-safe_eigvals * float(exp_time))

        if truncate_rank_ratio is not None:
            r = float(truncate_rank_ratio)
            assert 0.0 < r <= 1.0, r
            K = max(1, int((D2 * r + 0.999) // 1))
            if K < D2:
                trunc = torch.zeros_like(safe_eigvals)
                trunc[D2 - K:] = 1.0
                filter_vals = filter_vals * trunc

        combo_div = filter_vals / safe_eigvals
        DV = D_mat @ V_C
        DH = (DV * combo_div.unsqueeze(0)) @ V_C.T

        if init_mode == "zero":
            theta = DH
        else:
            if init_mode == "mean":
                theta_0 = v_stack.mean(dim=0)
            else:  # "sum"
                theta_0 = v_stack.sum(dim=0)
            T_G = (theta_0 @ V_C * filter_vals.unsqueeze(0)) @ V_C.T
            theta = theta_0 - T_G + DH

        return theta.detach().to(original_dtype)

    merged = {}
    name_list = list(task_vectors[0].keys())
    iterator = tqdm(name_list, desc="swudi", leave=False) if progress else name_list
    for name in iterator:
        param = task_vectors[0][name]
        if _is_2d_optimizable(param, name):
            v_stack = torch.stack([tv[name] for tv in task_vectors])
            merged[name] = _solve(name, v_stack).cpu()
        else:
            merged[name] = _incremental_average([tv[name] for tv in task_vectors])

    return _combine_with_pretrained(base_model, merged, float(scaling_coefficient))


# ---------------------------------------------------------------------------
# ASWUDI -- per-layer adaptive rank selection (zero continuous hyperparam)
# ---------------------------------------------------------------------------

def _aswudi_pick_rank(eigvals_desc: torch.Tensor, D1: int, D2: int,
                      method: str, min_rank: int, max_rank_ratio: float) -> int:
    K = eigvals_desc.numel()
    assert K == D2

    if method == "none":
        k = K
    elif method == "stable_rank":
        num = (eigvals_desc * eigvals_desc).sum()
        den = eigvals_desc[0].square().clamp(min=1e-30)
        k = int(torch.ceil(num / den).item())
    elif method == "entropy":
        p = (eigvals_desc / eigvals_desc.sum().clamp(min=1e-30)).clamp(min=1e-30)
        H = -(p * p.log()).sum()
        k = int(torch.ceil(H.exp()).item())
    elif method == "participation":
        s1 = eigvals_desc.sum()
        s2 = (eigvals_desc * eigvals_desc).sum().clamp(min=1e-30)
        k = int(torch.ceil((s1 * s1) / s2).item())
    elif method == "participation_sqrt":
        y = eigvals_desc.sqrt()
        s1 = y.sum()
        s2 = (y * y).sum().clamp(min=1e-30)
        k = int(torch.ceil((s1 * s1) / s2).item())
    elif method == "gavish_donoho":
        beta = float(min(D1, D2)) / float(max(D1, D2))
        y = eigvals_desc.sqrt()
        y_med = y.median().clamp(min=1e-30)
        if beta < 0.9999:
            mu_sq = (1.0 - beta) ** 2 + beta
            mu_beta = max(mu_sq, 1.0e-3) ** 0.5
        else:
            mu_beta = (2.0 + 3.0 ** 0.5) ** 0.5
        lam_star_coef = (
            (2.0 * (beta + 1.0))
            + (8.0 * beta) / ((beta + 1.0) + (beta * beta + 14.0 * beta + 1.0) ** 0.5)
        ) ** 0.5
        omega_beta = lam_star_coef / max(mu_beta, 1.0e-3)
        thresh = omega_beta * y_med.item()
        k = int((y > thresh).sum().item())
        k = max(k, 1)
    elif method.startswith("cumvar_"):
        frac = float(method.split("_", 1)[1])
        cum = eigvals_desc.cumsum(0) / eigvals_desc.sum().clamp(min=1e-30)
        k = int((cum < frac).sum().item()) + 1
    else:
        raise ValueError(f"Unknown auto_rank_method={method!r}")

    max_k = max(min_rank, int(round(max_rank_ratio * K)))
    return max(min_rank, min(k, max_k))


def aswudi_merge(
    base_model: nn.Module,
    finetuned_models: List[nn.Module],
    exclude_param_names_regex: Iterable[str],
    *,
    scaling_coefficient: float = 1.0,
    merge_device: Optional[str] = "cuda",
    auto_rank_method: str = "participation_sqrt",
    min_rank: int = 1,
    max_rank_ratio: float = 1.0,
    rank_scale: float = 1.0,
    filter_type: str = "none",
    exp_time: float = 1.0e6,
    init_mode: str = "sum",
    min_eigenvalue: float = 1.0e-8,
    progress: bool = True,
) -> dict:
    """ASWUDI: SWUDI with zero-hyperparameter per-layer adaptive rank.

    Default ``auto_rank_method='participation_sqrt'`` matches the paper's
    zero-hyperparameter recommendation for CLIP and Flan-T5 LoRA. For
    Llama-style full-parameter LLM/VLM deltas, set
    ``auto_rank_method='gavish_donoho'`` (best Llama-tuned rule).

    ``init_mode``:
      - "sum"  (default, matches CLIP / Flan-T5 / Llama paper): theta_0 = sum_i tau_i.
        Output delta has ~N x tau_i magnitude and the user must compensate at the
        outer scaling_coefficient (e.g. ~1/N on Qwen2-VL: scale=0.2 for N=5).
      - "mean": theta_0 = (1/N) sum_i tau_i. Output delta has ~1 x tau_i
        magnitude, mirroring OptMerge / wudi2 after Adam contraction. Lets
        scaling_coefficient=1.0 work directly without per-setup sweeps.
        Mathematically: aswudi(mean, scale=s) == aswudi(sum, scale=s/N) when
        the spectral filter is rank-only ('none'); with exp-filter this is
        slightly different but still off by an O(1) factor.
      - "zero": theta_0 = 0; merged delta is purely the spectral residual DH.
    """
    assert filter_type in ("none", "exponential"), filter_type
    assert init_mode in ("sum", "mean", "zero"), init_mode

    task_vectors = _build_task_vectors(
        base_model, finetuned_models, exclude_param_names_regex
    )
    device = merge_device
    per_layer_ranks: dict = {}

    def _solve(name, v_stack):
        original_dtype = v_stack.dtype
        target = device if device is not None else v_stack.device
        v_stack = v_stack.to(device=target, dtype=torch.float32)
        N, D1, D2 = v_stack.shape
        flat = v_stack.reshape(N, -1)
        l2_sq = torch.norm(flat, p=2, dim=-1).square()
        if l2_sq.max().item() < 1.0e-12:
            return _incremental_average(list(v_stack)).to(original_dtype)
        inv_norms = 1.0 / l2_sq.clamp(min=1.0e-12)

        B = torch.einsum("nab,nac->nbc", v_stack, v_stack)
        C = torch.einsum("n,nbc->bc", inv_norms, B)
        vB = torch.einsum("nab,nbc->nac", v_stack, B)
        D_mat = torch.einsum("n,nac->ac", inv_norms, vB)
        del B, vB

        eigvals_asc, V_asc = torch.linalg.eigh(C)
        safe_eigvals_asc = eigvals_asc.clamp(min=min_eigenvalue)
        eigvals_desc = safe_eigvals_asc.flip(0)

        K = _aswudi_pick_rank(
            eigvals_desc, D1=D1, D2=D2,
            method=auto_rank_method,
            min_rank=min_rank, max_rank_ratio=max_rank_ratio,
        )
        if abs(rank_scale - 1.0) > 1e-8:
            K = max(min_rank, min(D2, int(round(K * float(rank_scale)))))
        per_layer_ranks[name] = K

        filter_vals = torch.zeros_like(safe_eigvals_asc)
        if K > 0:
            if filter_type == "none":
                filter_vals[D2 - K:] = 1.0
            else:
                exp_part = 1.0 - torch.exp(-safe_eigvals_asc[D2 - K:] * float(exp_time))
                filter_vals[D2 - K:] = exp_part.clamp(min=0.0, max=1.0)

        combo_div = filter_vals / safe_eigvals_asc
        DV = D_mat @ V_asc
        DH = (DV * combo_div.unsqueeze(0)) @ V_asc.T

        if init_mode == "zero":
            theta = DH
        else:
            if init_mode == "mean":
                theta_0 = v_stack.mean(dim=0)
            else:  # "sum"
                theta_0 = v_stack.sum(dim=0)
            T_G = (theta_0 @ V_asc * filter_vals.unsqueeze(0)) @ V_asc.T
            theta = theta_0 - T_G + DH

        return theta.detach().to(original_dtype)

    merged = {}
    name_list = list(task_vectors[0].keys())
    iterator = tqdm(name_list, desc="aswudi", leave=False) if progress else name_list
    for name in iterator:
        param = task_vectors[0][name]
        if _is_2d_optimizable(param, name):
            v_stack = torch.stack([tv[name] for tv in task_vectors])
            merged[name] = _solve(name, v_stack).cpu()
        else:
            merged[name] = _incremental_average([tv[name] for tv in task_vectors])

    if per_layer_ranks:
        ranks = list(per_layer_ranks.values())
        ratios = [
            per_layer_ranks[n] / max(task_vectors[0][n].shape[-1], 1)
            for n in per_layer_ranks
        ]
        mean_r = sum(ratios) / len(ratios)
        srt = sorted(ranks)
        print(
            f"[aswudi/{auto_rank_method}] {len(ranks)} 2-D layers, "
            f"mean(K/D2)={mean_r:.3f}, "
            f"rank min={min(ranks)} median={srt[len(srt)//2]} max={max(ranks)}"
        )

    return _combine_with_pretrained(base_model, merged, float(scaling_coefficient))
