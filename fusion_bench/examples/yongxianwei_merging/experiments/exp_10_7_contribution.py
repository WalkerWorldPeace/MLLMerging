"""§10.7 — Spectral contribution decomposition (proxy version).

For each layer of CLIP-ViT-B/32 TA8, we measure how the WUDI proxy
    P(θ) = Σ_i ||(θ - v_i) v_i^T||² / ||v_i||²
changes as we use only a subset of eigen-directions to compute θ. We define
    θ_K = θ_0 + B V_K diag(1/λ_K) V_K^T   (top-K hard-truncation)
and compute the proxy P(θ_K) for K ∈ {1, ..., d}. The "marginal contribution"
of direction k is P(θ_{k-1}) - P(θ_k).

Theory claim (§10.7): heads decrease proxy strongly, tails contribute little
or even *increase* downstream interference (since they amplify noise).

Note
----
This experiment uses the WUDI proxy, NOT real downstream accuracy, since the
latter requires running CLIP forward over actual datasets. Per the user's
guidance we report a separate "downstream proxy" number using the layer-level
proxy reduction. The full real-accuracy version (with image evaluation) is
§10.7-real and is left as a manual experiment.

Output:
    outputs/yongxianwei_merging/theory_diagnostics/exp_10_7_contribution/
        per_layer.json   (proxy_curve, marginal_curve)
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
from _common import (DEFAULT_OUT, compute_C_D, eigh_descending,
                     hard_truncation_theta, write_json,
                     configure_torch_for_diagnostics,
                     participation_sqrt_rank)


def wudi_proxy(theta: torch.Tensor, v_stack: torch.Tensor) -> float:
    """Σ_i ||(θ - v_i) v_i^T||² / ||v_i||²."""
    eps = 1e-12
    N = v_stack.shape[0]
    flat = v_stack.reshape(N, -1)
    l2 = (flat * flat).sum(dim=-1).clamp(min=eps)
    diff = theta.unsqueeze(0) - v_stack                      # (N, m, d)
    prod = torch.matmul(diff, v_stack.transpose(1, 2))       # (N, m, m)
    per = prod.square().sum(dim=(1, 2)) / l2                 # (N,)
    return float(per.sum().item())


def cumulative_proxy_curve(v_stack: torch.Tensor, lam: torch.Tensor, V: torch.Tensor,
                           D: torch.Tensor, theta0: torch.Tensor,
                           K_values: List[int]) -> List[float]:
    out = []
    for K in K_values:
        theta_K = hard_truncation_theta(theta0, D, lam, V, K)
        out.append(wudi_proxy(theta_K, v_stack))
    return out


def diagnose_layer(name: str, v_stack: torch.Tensor) -> Dict:
    v_stack = v_stack.to(torch.float32)
    C, D, l2_sq, theta0 = compute_C_D(v_stack)
    lam, V = eigh_descending(C)
    N, m, d = v_stack.shape

    # Sample K values logarithmically + key markers
    K_set = set([0])
    for f in (0.01, 0.025, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0):
        K_set.add(max(1, int(round(f * d))))
    K_aswudi = participation_sqrt_rank(lam.cpu())
    K_set.add(K_aswudi)
    K_values = sorted(K_set)

    proxy_curve = cumulative_proxy_curve(v_stack, lam, V, D, theta0, K_values)
    proxy_at_0 = proxy_curve[0]
    proxy_at_d = proxy_curve[-1]

    # Marginal effect of going from K_(i-1) to K_i: positive = proxy reduced
    # (good); negative = proxy increased (bad direction added).
    marginals = []
    for i in range(1, len(K_values)):
        K_prev, K_now = K_values[i-1], K_values[i]
        delta = proxy_curve[i-1] - proxy_curve[i]   # reduction
        marginals.append({"from": K_prev, "to": K_now, "reduction": delta})

    # Bin head/mid/tail and compare reductions
    head_K = max(1, int(round(0.10 * d)))
    mid_K = max(head_K, int(round(0.50 * d)))

    th_head = hard_truncation_theta(theta0, D, lam, V, head_K)
    th_mid = hard_truncation_theta(theta0, D, lam, V, mid_K)
    th_aswudi = hard_truncation_theta(theta0, D, lam, V, K_aswudi)
    th_full = hard_truncation_theta(theta0, D, lam, V, d)
    proxy_head = wudi_proxy(th_head, v_stack)
    proxy_mid = wudi_proxy(th_mid, v_stack)
    proxy_aswudi = wudi_proxy(th_aswudi, v_stack)
    proxy_full = wudi_proxy(th_full, v_stack)
    proxy_zero = wudi_proxy(theta0, v_stack)

    return {
        "layer": name,
        "shape": [m, d],
        "n_experts": N,
        "K_aswudi": K_aswudi,
        "proxy_curve_K": K_values,
        "proxy_curve_P": proxy_curve,
        "marginals": marginals,
        # Compact summary
        "proxy_zero": proxy_zero,           # K=0
        "proxy_full": proxy_full,           # K=d (closed-form on top-d)
        "proxy_aswudi": proxy_aswudi,
        "proxy_head_10pct": proxy_head,
        "proxy_mid_50pct": proxy_mid,
        # Did adding tail directions REDUCE or INCREASE proxy?
        "tail_reduction": proxy_aswudi - proxy_full,   # positive = adding tail helps reduce more
        "tail_relative": (proxy_aswudi - proxy_full) / max(proxy_zero, 1e-30),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "exp_10_7_contribution"))
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
    for i, name in enumerate(layer_names):
        v = blob["stacks"][name].to(torch.float32).to(device)
        result = diagnose_layer(name, v)
        rows.append(result)
        if (i + 1) % 8 == 0:
            print(f"  [10.7] {i+1}/{len(layer_names)}  "
                  f"latest tail_relative={result['tail_relative']:.4f}", flush=True)

    write_json(out_dir / "per_layer.json", rows)

    import statistics as st
    summary = {
        "n_layers": len(rows),
        "tail_relative_mean": st.fmean([r["tail_relative"] for r in rows]),
        "tail_relative_median": st.median([r["tail_relative"] for r in rows]),
        "tail_relative_min": min(r["tail_relative"] for r in rows),
        "tail_relative_max": max(r["tail_relative"] for r in rows),
        # If tail_relative > 0 it means ASWUDI's K leaves more proxy than closed-form
        # (truncation increases proxy — expected since proxy is the WUDI loss).
        "fraction_proxy_drops_with_full": sum(
            1 for r in rows if r["proxy_full"] < r["proxy_aswudi"]
        ) / len(rows),
        "proxy_full_mean": st.fmean([r["proxy_full"] for r in rows]),
        "proxy_aswudi_mean": st.fmean([r["proxy_aswudi"] for r in rows]),
        "proxy_zero_mean": st.fmean([r["proxy_zero"] for r in rows]),
        "proxy_head_10pct_mean": st.fmean([r["proxy_head_10pct"] for r in rows]),
        "proxy_mid_50pct_mean": st.fmean([r["proxy_mid_50pct"] for r in rows]),
        # How much does cumulative proxy drop in head vs tail?
        # Compute (P(0) - P(K_aswudi)) / (P(0) - P(d))
        "fraction_head_explains": st.fmean([
            (r["proxy_zero"] - r["proxy_aswudi"]) / max(r["proxy_zero"] - r["proxy_full"], 1e-12)
            for r in rows
        ]),
    }
    write_json(out_dir / "summary.json", summary)
    print("[10.7] summary:")
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
