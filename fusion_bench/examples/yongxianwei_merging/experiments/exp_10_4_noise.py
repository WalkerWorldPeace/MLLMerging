"""§10.4 — Spectral noise-model diagnostic.

For each layer's WUDI normal equation
    Bq_k = λ_k R_k° + ξ_k,    B = D - θ_0 C
we want to test whether the implied per-direction noise ν_k² := E ||ξ_k||²/d_o
amplifies in the small-λ tail (i.e. ν_k²/λ_k² grows as λ_k → 0). Since R_k°
is unobservable, we estimate ξ_k via two complementary methods:

  (a) Leave-one-task-out bootstrap. Compute D⁽⁻i⁾ = (D - v_i A_i) and
      C⁽⁻i⁾ = (C - A_i). Project Dq_k - λ_k * (closed-form θ_(-i) q_k)
      onto V's eigenbasis. The variance across i estimates ν_k².

  (b) Closed-form residual. Define R̂_k = (Dq_k)/λ_k - θ_0 q_k for λ_k > eps.
      Compare ||R̂_k|| against ||θ_0 q_k|| as a function of k. The tail
      should diverge if ν_k²/λ_k² grows.

We also fit ν_k² = σ_0² + σ_1² λ_k^α and report α (the prediction is α<2,
which makes ν_k²/λ_k² → ∞ in the small-λ tail).

Output:
    outputs/yongxianwei_merging/theory_diagnostics/exp_10_4_noise/
        per_layer.json
        summary.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (DEFAULT_OUT, compute_C_D, eigh_descending, write_json,
                     configure_torch_for_diagnostics)


def loo_bootstrap_xi(v_stack: torch.Tensor, V: torch.Tensor,
                     lam: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
    """Leave-one-out estimate of ν_k² across N experts.

    Returns
    -------
    nu_sq : (d,) — sample variance (across i) of ||(D⁽⁻i⁾ - λ_k * R̂⁽⁻i⁾_k) q_k||² / d_o
            where R̂⁽⁻i⁾_k = (D⁽⁻i⁾ q_k) / λ_k° (use λ from full C; bias-corrected)
    """
    N, m, d = v_stack.shape
    eps = 1e-12
    flat = v_stack.reshape(N, -1)
    l2_sq = (flat * flat).sum(dim=-1).clamp(min=eps)        # (N,)
    inv = 1.0 / l2_sq                                      # (N,)
    Bs = torch.einsum("nab,nac->nbc", v_stack, v_stack)    # (N, d, d)
    A_full = torch.einsum("n,nbc->bc", inv, Bs)            # = C
    D_full_full = torch.einsum("n,nab,nbc->ac", inv, v_stack, Bs)
    # Per-expert contribution:
    A_i = inv.view(N, 1, 1) * Bs                           # (N, d, d)
    D_i_contrib = torch.einsum("n,nab,nbc->nac", inv, v_stack, Bs)  # (N, m, d)

    # For each i, compute D⁽⁻i⁾ q_k = (D - D_i_contrib_i) V and λ_full q_k
    # and a "leave-one-out closed form" residual.
    Dq_full = D_full_full @ V                               # (m, d)
    samples = []
    for i in range(N):
        Dq_loo = Dq_full - D_i_contrib[i] @ V                # (m, d)
        # Use full λ as denominator (more stable proxy)
        # The residual in eigenbasis is Dq_loo / λ - θ0 q_k = ξ_k^(-i) / λ_k + R°_k
        # We approximate ξ_k by the variation around the mean across i:
        samples.append(Dq_loo)
    # Stack to (N, m, d)
    Dq_stack = torch.stack(samples, dim=0)
    # Variance per (m, d) location across N experts
    mean_Dq = Dq_stack.mean(dim=0)                          # (m, d)
    var_Dq = ((Dq_stack - mean_Dq) ** 2).mean(dim=0)        # (m, d)
    # Direction-wise mean over output dim m
    nu_sq = var_Dq.mean(dim=0)                              # (d,)
    return nu_sq, Dq_full, mean_Dq


def fit_nu_model(lam: torch.Tensor, nu_sq: torch.Tensor):
    """Fit ν² = σ0² + σ1² λ^α.  Robust to λ=0 directions (drop them)."""
    mask = (lam > 1e-8) & (nu_sq > 0)
    if mask.sum() < 5:
        return {"alpha": float("nan"), "sigma0": float("nan"), "sigma1": float("nan")}
    log_lam = torch.log(lam[mask])
    log_nu = torch.log(nu_sq[mask])
    # Initial fit: log nu² ≈ log(sigma1²) + α log λ  (assume σ0² small, drop intercept)
    n = log_lam.numel()
    a = log_lam
    b = log_nu
    am = a.mean(); bm = b.mean()
    slope = float(((a - am) * (b - bm)).sum() / ((a - am) ** 2).sum().clamp(min=1e-30))
    intercept = float(bm - slope * am)
    return {
        "alpha": slope,
        "log_sigma1_sq": intercept,
        "sigma1_sq_at_lam_max": math.exp(intercept + slope * float(log_lam.max().item())),
        "sigma1_sq_at_lam_min": math.exp(intercept + slope * float(log_lam.min().item())),
    }


def diagnose_layer(name: str, v_stack: torch.Tensor) -> Dict:
    v_stack = v_stack.to(torch.float32)
    C, D, l2_sq, theta0 = compute_C_D(v_stack)
    lam, V = eigh_descending(C)
    N, m, d = v_stack.shape

    nu_sq, Dq_full, mean_Dq = loo_bootstrap_xi(v_stack, V, lam, theta0)
    nu_over_lam_sq = nu_sq / lam.clamp(min=1e-30) ** 2

    # Closed-form residual ||(Dq_k)/λ_k|| direction profile
    cf_dir_norm = (Dq_full / lam.clamp(min=1e-30).unsqueeze(0)).norm(dim=0)
    theta0_dir_norm = (theta0 @ V).norm(dim=0)

    # Fit
    fit = fit_nu_model(lam, nu_sq)

    # Report dims trimmed to 256 head + 256 tail to keep file small
    keep = min(d, 256)
    return {
        "layer": name,
        "shape": [m, d],
        "n_experts": N,
        "lam": lam[:keep].cpu().tolist() + lam[-keep:].cpu().tolist() if d > 2*keep else lam.cpu().tolist(),
        "nu_sq": nu_sq[:keep].cpu().tolist() + nu_sq[-keep:].cpu().tolist() if d > 2*keep else nu_sq.cpu().tolist(),
        "nu_over_lam_sq": nu_over_lam_sq[:keep].cpu().tolist() + nu_over_lam_sq[-keep:].cpu().tolist() if d > 2*keep else nu_over_lam_sq.cpu().tolist(),
        "cf_dir_norm": cf_dir_norm[:keep].cpu().tolist() + cf_dir_norm[-keep:].cpu().tolist() if d > 2*keep else cf_dir_norm.cpu().tolist(),
        "theta0_dir_norm": theta0_dir_norm[:keep].cpu().tolist() + theta0_dir_norm[-keep:].cpu().tolist() if d > 2*keep else theta0_dir_norm.cpu().tolist(),
        "head_tail": (d > 2*keep),  # if true, lists are 256 head + 256 tail
        "head_tail_keep": keep,
        "fit": fit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "exp_10_4_noise"))
    ap.add_argument("--task_vectors", default=str(DEFAULT_OUT / "task_vectors.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_layers", type=int, default=None,
                    help="If set, only diagnose this many layers (for smoke runs).")
    args = ap.parse_args()

    configure_torch_for_diagnostics()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.task_vectors, map_location="cpu", weights_only=False)
    layer_names = blob["layer_names"]
    if args.max_layers is not None:
        layer_names = layer_names[:args.max_layers]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = []
    alpha_vals = []
    for i, name in enumerate(layer_names):
        v = blob["stacks"][name].to(torch.float32).to(device)
        result = diagnose_layer(name, v)
        rows.append(result)
        alpha_vals.append(result["fit"].get("alpha", float("nan")))
        if (i + 1) % 8 == 0:
            print(f"  [10.4] {i+1}/{len(layer_names)} (latest α={alpha_vals[-1]:.3f})", flush=True)

    write_json(out_dir / "per_layer.json", rows)
    import statistics as st
    valid_alpha = [a for a in alpha_vals if not (a is None or math.isnan(a))]
    summary = {
        "n_layers": len(rows),
        "alpha_mean": st.fmean(valid_alpha) if valid_alpha else float("nan"),
        "alpha_median": st.median(valid_alpha) if valid_alpha else float("nan"),
        "alpha_min": min(valid_alpha) if valid_alpha else float("nan"),
        "alpha_max": max(valid_alpha) if valid_alpha else float("nan"),
        "alpha_stdev": st.stdev(valid_alpha) if len(valid_alpha) > 1 else 0.0,
        "fraction_alpha_lt_2": sum(a < 2 for a in valid_alpha) / max(len(valid_alpha), 1),
        "fraction_alpha_lt_1": sum(a < 1 for a in valid_alpha) / max(len(valid_alpha), 1),
        "fraction_alpha_lt_0": sum(a < 0 for a in valid_alpha) / max(len(valid_alpha), 1),
    }
    write_json(out_dir / "summary.json", summary)
    print("[10.4] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import json
    main()
