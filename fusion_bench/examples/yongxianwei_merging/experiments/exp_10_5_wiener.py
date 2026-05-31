"""§10.5 — Wiener vs hard truncation predicted-risk calibration.

Using the per-layer τ̂_k², ν̂_k² estimates from §10.4 (and a τ̂_k² proxy
derived from the residual ||(Dq_k)/λ_k - θ_0 q_k|| profile), we form an
empirical Wiener filter
    h_k* = ρ_k / (1 + ρ_k),  ρ_k = λ_k² τ̂_k² / ν̂_k²
and compare its **predicted Bayes risk**
    Σ_k d_o · [(1 - h_k)² τ_k² + h_k² ν_k²/λ_k²]
against:
    - closed-form (h=1)
    - IWUDI exponential (h = 1 - exp(-λ t)) at the WUDI tuned t = 300
    - SWUDI hard at uniform K = ⌈0.65 d⌉
    - ASWUDI hard at K = K_{√λ} per layer

Limitation
----------
We do not estimate τ_k² directly: as in §10.4, only ν̂_k² is observable
through LOO bootstrap. We use **τ̂_k² := ||(Dq_k)/λ_k - θ_0 q_k||²/d_o**
as the residual-magnitude proxy. This is biased upward by ν̂_k²/λ_k² (the
proxy is the noise-contaminated residual), so the resulting "predicted
risk" is a worst-case Bayes risk under the τ proxy. We document this
caveat in the JSON output and in the theory_framework.md update.

Output:
    outputs/yongxianwei_merging/theory_diagnostics/exp_10_5_wiener/
        per_layer.json
        summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (DEFAULT_OUT, compute_C_D, eigh_descending, write_json,
                     filter_to_theta, configure_torch_for_diagnostics,
                     participation_sqrt_rank)


def loo_nu_sq(v_stack: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Leave-one-out variance of D q_k across N experts (per-direction ν̂²)."""
    N, m, d = v_stack.shape
    eps = 1e-12
    flat = v_stack.reshape(N, -1)
    l2_sq = (flat * flat).sum(dim=-1).clamp(min=eps)
    inv = 1.0 / l2_sq
    Bs = torch.einsum("nab,nac->nbc", v_stack, v_stack)
    D_full = torch.einsum("n,nab,nbc->ac", inv, v_stack, Bs)
    D_i_contrib = torch.einsum("n,nab,nbc->nac", inv, v_stack, Bs)

    Dq_full = D_full @ V                                    # (m, d)
    losses = []
    for i in range(N):
        Dq_loo = (D_full - D_i_contrib[i]) @ V              # (m, d)
        losses.append(Dq_loo)
    Dq_stack = torch.stack(losses, dim=0)                   # (N, m, d)
    mean_Dq = Dq_stack.mean(dim=0)
    var_Dq = ((Dq_stack - mean_Dq) ** 2).mean(dim=0)        # (m, d)
    return var_Dq.mean(dim=0), Dq_full


def predicted_risk(h: torch.Tensor, lam: torch.Tensor,
                   tau_sq: torch.Tensor, nu_sq: torch.Tensor,
                   d_o: int) -> float:
    """Σ_k d_o [ (1-h_k)² τ_k² + h_k² ν_k² / λ_k² ]."""
    eps = 1e-30
    nu_over = nu_sq / lam.clamp(min=eps).pow(2)
    nu_over = torch.where(lam > 1e-12, nu_over, torch.zeros_like(nu_over))
    bias = (1 - h) ** 2 * tau_sq
    var = h ** 2 * nu_over
    return float(d_o * (bias + var).sum().item())


def diagnose_layer(name: str, v_stack: torch.Tensor) -> Dict:
    v_stack = v_stack.to(torch.float32)
    C, D, l2_sq, theta0 = compute_C_D(v_stack)
    lam, V = eigh_descending(C)
    N, m, d = v_stack.shape

    nu_sq, Dq_full = loo_nu_sq(v_stack, V)
    eps_lam = 1e-12
    # τ̂ = ||(Dq_k)/λ_k - θ_0 q_k|| / sqrt(m)
    R_hat = Dq_full / lam.clamp(min=eps_lam).unsqueeze(0)   # (m, d) ≈ θ° q_k
    theta0V = theta0 @ V                                     # (m, d)
    R_residual = R_hat - theta0V                             # (m, d)
    tau_sq = (R_residual * R_residual).mean(dim=0)           # (d,)

    rho = lam.pow(2) * tau_sq / nu_sq.clamp(min=1e-30)
    h_wiener = rho / (1 + rho)

    # Filters
    h_cf = (lam > eps_lam).float()
    h_iwudi = 1.0 - torch.exp(-300.0 * lam.clamp(min=0))
    K_swudi = max(1, int(math.ceil(0.65 * d)))
    h_swudi = torch.zeros_like(lam)
    h_swudi[:K_swudi] = 1.0
    K_aswudi = participation_sqrt_rank(lam.cpu())
    h_aswudi = torch.zeros_like(lam)
    h_aswudi[:K_aswudi] = 1.0

    risks = {
        "closed_form": predicted_risk(h_cf, lam, tau_sq, nu_sq, m),
        "iwudi_t300": predicted_risk(h_iwudi, lam, tau_sq, nu_sq, m),
        "swudi_r0_65": predicted_risk(h_swudi, lam, tau_sq, nu_sq, m),
        "aswudi_sqrt": predicted_risk(h_aswudi, lam, tau_sq, nu_sq, m),
        "wiener": predicted_risk(h_wiener, lam, tau_sq, nu_sq, m),
        "drop_all": predicted_risk(torch.zeros_like(lam), lam, tau_sq, nu_sq, m),
    }

    # SNR knee location: smallest k with rho < 1
    knee = int((rho < 1).float().argmax().item()) if (rho < 1).any() else d
    boundary_rho = float(rho[K_aswudi - 1].item()) if 0 < K_aswudi <= d else float("nan")
    boundary_rho_next = float(rho[K_aswudi].item()) if 0 <= K_aswudi < d else float("nan")

    keep = min(d, 256)
    hd = (d > 2*keep)
    return {
        "layer": name,
        "shape": [m, d],
        "n_experts": N,
        "K_aswudi": K_aswudi,
        "K_swudi": K_swudi,
        "knee_index_rho_lt_1": knee,
        "boundary_rho_at_K": boundary_rho,
        "boundary_rho_after_K": boundary_rho_next,
        "predicted_risk": risks,
        "lam_head": lam[:keep].cpu().tolist(),
        "rho_head": rho[:keep].cpu().tolist(),
        "rho_tail": rho[-keep:].cpu().tolist() if hd else [],
        "h_wiener_head": h_wiener[:keep].cpu().tolist(),
        "head_tail_keep": keep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "exp_10_5_wiener"))
    ap.add_argument("--task_vectors", default=str(DEFAULT_OUT / "task_vectors.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_layers", type=int, default=None)
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
    risk_by_method: Dict[str, List[float]] = {}
    for i, name in enumerate(layer_names):
        v = blob["stacks"][name].to(torch.float32).to(device)
        result = diagnose_layer(name, v)
        rows.append(result)
        for k, val in result["predicted_risk"].items():
            risk_by_method.setdefault(k, []).append(val)
        if (i + 1) % 8 == 0:
            print(f"  [10.5] {i+1}/{len(layer_names)}", flush=True)

    write_json(out_dir / "per_layer.json", rows)

    import statistics as st
    summary = {
        "n_layers": len(rows),
        "method_risk_summary": {
            k: {
                "total": sum(v),
                "mean": st.fmean(v),
                "median": st.median(v),
                "max": max(v),
            } for k, v in risk_by_method.items()
        },
        # Per-layer: which method has the lowest predicted risk?
        "argmin_count": {
            k: sum(1 for j in range(len(rows))
                   if min(rows[j]["predicted_risk"], key=lambda x: rows[j]["predicted_risk"][x]) == k)
            for k in risk_by_method.keys()
        },
        # Boundary SNR (does ASWUDI cut at a low-ρ region?)
        "boundary_rho_at_K_mean": st.fmean([r["boundary_rho_at_K"] for r in rows
                                            if not math.isnan(r["boundary_rho_at_K"])]),
        "boundary_rho_at_K_median": st.median([r["boundary_rho_at_K"] for r in rows
                                                if not math.isnan(r["boundary_rho_at_K"])]),
        "boundary_rho_after_K_median": st.median([r["boundary_rho_after_K"] for r in rows
                                                   if not math.isnan(r["boundary_rho_after_K"])]),
        "fraction_K_below_knee": sum(1 for r in rows
                                     if r["K_aswudi"] <= r["knee_index_rho_lt_1"]) / len(rows),
    }
    write_json(out_dir / "summary.json", summary)
    print("[10.5] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
