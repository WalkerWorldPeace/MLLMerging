"""§10.6 — Rank-rule spectrum visualization.

For each 2-D layer of CLIP-ViT-B/32 TA8, compute the eigenspectrum of C and
the rank K selected by:
    - K_{√λ} : participation_sqrt
    - K_λ    : participation
    - K_GD   : Gavish-Donoho 2014 hard threshold
We compare these against ASWUDI's actual selected rank (from a real run on
this pool — the rank distribution is recorded in the layer-level rank cache,
or we can simply call participation_sqrt as the de-facto reference).

Because we don't run a fresh global SWUDI grid sweep here, we rely on the
already-tuned SWUDI optimum (r=0.65 on TA8) as the "best uniform K" reference;
this is documented in §1 of theory_framework.md.

Output:
    outputs/yongxianwei_merging/theory_diagnostics/exp_10_6_rank/
        per_layer.json   (lam, K_sqrt, K_lambda, K_GD, K_swudi65)
        rule_agreement.json
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
from _common import (DEFAULT_OUT, compute_C_D, eigh_descending,
                     participation_sqrt_rank, participation_rank,
                     gavish_donoho_rank, write_json,
                     configure_torch_for_diagnostics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "exp_10_6_rank"))
    ap.add_argument("--task_vectors", default=str(DEFAULT_OUT / "task_vectors.pt"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    configure_torch_for_diagnostics()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.task_vectors, map_location="cpu", weights_only=False)
    layer_names = blob["layer_names"]
    print(f"[10.6] {len(layer_names)} layers")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = []
    for name in layer_names:
        v_stack = blob["stacks"][name].to(torch.float32).to(device)
        C, D, l2_sq, theta0 = compute_C_D(v_stack)
        lam, V = eigh_descending(C)
        m, d = v_stack.shape[1], v_stack.shape[2]
        N = v_stack.shape[0]

        K_sqrt = participation_sqrt_rank(lam.cpu())
        K_lam = participation_rank(lam.cpu())
        # GD: stack experts with frob-normalisation, that's how §8.1 defines M
        # Equivalent: s_k = √λ_k(C). Here m_eff = N·m (by construction of C).
        K_gd = gavish_donoho_rank(lam.cpu(), m=N * m, d=d)

        # SWUDI tuned uniform r=0.65 on TA8 (best-by-grid in our paper)
        K_swudi65 = int(math.ceil(0.65 * d))

        row = {
            "layer": name,
            "shape": [m, d],
            "n_experts": N,
            "K_sqrt": K_sqrt,
            "K_lambda": K_lam,
            "K_GD": K_gd,
            "K_swudi65": K_swudi65,
            "K_sqrt_ratio": K_sqrt / d,
            "K_lambda_ratio": K_lam / d,
            "K_GD_ratio": K_gd / d,
            "K_swudi65_ratio": K_swudi65 / d,
            "lambda": lam.cpu().tolist(),
        }
        rows.append(row)
        del v_stack, C, D, lam, V

    write_json(out_dir / "per_layer.json", rows)

    # Aggregate: rule agreement and per-shape distributions
    def stats(values):
        if not values: return {}
        import statistics as st
        return {
            "mean": st.fmean(values),
            "median": st.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": st.stdev(values) if len(values) > 1 else 0.0,
        }

    agreement = {
        "K_sqrt": stats([r["K_sqrt_ratio"] for r in rows]),
        "K_lambda": stats([r["K_lambda_ratio"] for r in rows]),
        "K_GD": stats([r["K_GD_ratio"] for r in rows]),
        "K_swudi65": stats([r["K_swudi65_ratio"] for r in rows]),
        # Pairwise rank correlations
        "corr_sqrt_vs_lambda": _corr([r["K_sqrt_ratio"] for r in rows],
                                     [r["K_lambda_ratio"] for r in rows]),
        "corr_sqrt_vs_GD": _corr([r["K_sqrt_ratio"] for r in rows],
                                 [r["K_GD_ratio"] for r in rows]),
        "corr_sqrt_vs_swudi65": _corr([r["K_sqrt_ratio"] for r in rows],
                                      [r["K_swudi65_ratio"] for r in rows]),
        "corr_GD_vs_swudi65": _corr([r["K_GD_ratio"] for r in rows],
                                    [r["K_swudi65_ratio"] for r in rows]),
    }
    write_json(out_dir / "rule_agreement.json", agreement)

    # By shape (different param types) – we sometimes care about MLP fc1 / out_proj
    by_kind: Dict[str, list] = {}
    for r in rows:
        key = _layer_kind(r["layer"])
        by_kind.setdefault(key, []).append(r)
    by_kind_summary = {}
    for k, rs in by_kind.items():
        by_kind_summary[k] = {
            "count": len(rs),
            "K_sqrt_mean": float(sum(r["K_sqrt_ratio"] for r in rs) / len(rs)),
            "K_lambda_mean": float(sum(r["K_lambda_ratio"] for r in rs) / len(rs)),
            "K_GD_mean": float(sum(r["K_GD_ratio"] for r in rs) / len(rs)),
        }
    write_json(out_dir / "by_kind.json", by_kind_summary)

    print("[10.6] rule agreement summary:")
    print(json.dumps(agreement, indent=2))


def _corr(xs, ys):
    """Pearson correlation."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    import statistics as st
    mx, my = st.fmean(xs), st.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _layer_kind(name: str) -> str:
    if "mlp.fc1" in name: return "mlp_fc1"
    if "mlp.fc2" in name: return "mlp_fc2"
    if "self_attn.q_proj" in name: return "attn_q"
    if "self_attn.k_proj" in name: return "attn_k"
    if "self_attn.v_proj" in name: return "attn_v"
    if "self_attn.out_proj" in name: return "attn_out"
    return "other"


if __name__ == "__main__":
    main()
