"""Re-aggregate and present the 3-scenario spectral diagnostic.

Reads per_layer.json produced by spectral_diagnostic.py, computes richer
layer-wise heterogeneity stats (CV of lambda_max, etc.), and prints a
side-by-side comparison against the 3 quantitative predictions in
iwudi_paper.md §5.10.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Dict, List

POOLS = [
    ("CLIP-ViT-B/32 TA8",       "clip_vit_b32_ta8"),
    ("Flan-T5 GLUE LoRA r=16",  "flan_t5_glue_lora16"),
    ("Llama-3.2-3B MergeBench", "llama32_3b"),
]
BASE = Path("outputs/yongxianwei_merging/spectral")


def _describe(vs: List[float]) -> Dict[str, float]:
    if not vs:
        return {}
    mean = st.fmean(vs)
    median = st.median(vs)
    stdev = st.stdev(vs) if len(vs) > 1 else 0.0
    out = {"mean": mean, "median": median, "stdev": stdev,
           "min": min(vs), "max": max(vs)}
    out["cv"] = stdev / mean if mean != 0 else float("nan")
    return out


def _summarise(per_layer: List[Dict]) -> Dict:
    def col(k): return [r[k] for r in per_layer if r.get(k) == r.get(k)]
    return {
        "n_layers": len(per_layer),
        "r_eff_norm": _describe(col("r_eff_norm")),
        "peak_ratio": _describe(col("peak_ratio")),
        "lambda_max": _describe(col("lambda_max")),
        "cumvar_at_0.50": _describe(col("cumvar_at_0.50")),
        "cumvar_at_0.65": _describe(col("cumvar_at_0.65")),
        "cumvar_at_0.85": _describe(col("cumvar_at_0.85")),
        "decay_tail_ratio": _describe(col("decay_tail_ratio")),
        "fro_ratio": _describe([r["mean_expert_fro"] / r["base_frob"]
                                for r in per_layer
                                if r.get("base_frob", 0) > 0 and r.get("mean_expert_fro", 0) > 0]),
    }


def main():
    results = {}
    for label, key in POOLS:
        per_layer = json.loads((BASE / key / "per_layer.json").read_text())
        results[label] = _summarise(per_layer)

    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "comparison.json").write_text(json.dumps(results, indent=2))

    def col_get(r, k1, k2):
        return r.get(k1, {}).get(k2, float("nan"))

    metrics = [
        ("n layers",                        "n_layers",        None,      "d"),
        ("r_eff_norm (mean)",              "r_eff_norm",      "mean",    ".4f"),
        ("r_eff_norm (median)",            "r_eff_norm",      "median",  ".4f"),
        ("peak_ratio λmax/λmedian (mean)", "peak_ratio",      "mean",    ".2e"),
        ("peak_ratio λmax/λmedian (median)","peak_ratio",     "median",  ".2e"),
        ("cumvar @ r=0.50 (mean)",         "cumvar_at_0.50",  "mean",    ".4f"),
        ("cumvar @ r=0.65 (mean)",         "cumvar_at_0.65",  "mean",    ".4f"),
        ("cumvar @ r=0.85 (mean)",         "cumvar_at_0.85",  "mean",    ".4f"),
        ("decay_tail λ_90/λmax (mean)",    "decay_tail_ratio","mean",    ".4f"),
        ("λmax CV (layer heterogeneity)",  "lambda_max",      "cv",      ".3f"),
        ("‖Δ‖/‖W‖ (mean)",                 "fro_ratio",       "mean",    ".4f"),
    ]

    hdr = f"{'metric':<42}  " + "  ".join(f"{label:>24s}" for label, _ in POOLS)
    print(hdr)
    print("-" * len(hdr))
    for disp, key, sub, fmt in metrics:
        vals = []
        for label, _ in POOLS:
            if sub is None:
                vals.append(results[label][key])
            else:
                vals.append(col_get(results[label], key, sub))
        if fmt == "d":
            s = "  ".join(f"{v:>24d}" for v in vals)
        else:
            s = "  ".join(f"{v:>24{fmt}}" for v in vals)
        print(f"{disp:<42}  {s}")

    print()
    print("=== 3 quantitative predictions (iwudi_paper.md §5.10, Δ-structure hypothesis) ===")
    r_eff_llama = col_get(results["Llama-3.2-3B MergeBench"], "r_eff_norm", "mean")
    r_eff_clip  = col_get(results["CLIP-ViT-B/32 TA8"], "r_eff_norm", "mean")
    peak_clip   = col_get(results["CLIP-ViT-B/32 TA8"], "peak_ratio", "median")
    peak_llama  = col_get(results["Llama-3.2-3B MergeBench"], "peak_ratio", "median")
    cv_clip     = col_get(results["CLIP-ViT-B/32 TA8"], "lambda_max", "cv")
    cv_llama    = col_get(results["Llama-3.2-3B MergeBench"], "lambda_max", "cv")
    print(f"P1 (effective rank): Llama / CLIP = {r_eff_llama/r_eff_clip:.2f}× "
          f"(predicted 3-5×)")
    print(f"   [Llama r_eff_norm = {r_eff_llama:.4f}, CLIP r_eff_norm = {r_eff_clip:.4f}]")
    print(f"P2 (spectral peak): CLIP / Llama (median peak_ratio) = {peak_clip/peak_llama:.2f}× "
          f"(predicted 2-3×)")
    print(f"   [CLIP peak = {peak_clip:.1f}, Llama peak = {peak_llama:.1f}]")
    print(f"P3 (layer heterogeneity): Llama λmax CV / CLIP λmax CV = {cv_llama/cv_clip:.2f}× "
          f"(predicted ~2×)")
    print(f"   [Llama CV = {cv_llama:.3f}, CLIP CV = {cv_clip:.3f}]")
    print()
    print("=== supplementary: cumvar at the hparam-optimum truncation ratio ===")
    print(f"CLIP  best r=0.65: cumvar = {col_get(results['CLIP-ViT-B/32 TA8'], 'cumvar_at_0.65', 'mean'):.4f}")
    print(f"Llama best r=0.85: cumvar = {col_get(results['Llama-3.2-3B MergeBench'], 'cumvar_at_0.85', 'mean'):.4f}")
    print(f"Llama AT r=0.65:   cumvar = {col_get(results['Llama-3.2-3B MergeBench'], 'cumvar_at_0.65', 'mean'):.4f}  "
          "<-- if CLIP's optimal r were applied to Llama, this much variance would be kept")
    print(f"CLIP  AT r=0.85:   cumvar = {col_get(results['CLIP-ViT-B/32 TA8'], 'cumvar_at_0.85', 'mean'):.4f}  "
          "<-- if Llama's optimal r were applied to CLIP, this much variance would be kept")
    print()
    print("=== Flan-T5 LoRA extreme ===")
    print(f"Flan-T5 peak_ratio median = {col_get(results['Flan-T5 GLUE LoRA r=16'], 'peak_ratio', 'median'):.2e}")
    print(f"Flan-T5 r_eff_norm  mean  = {col_get(results['Flan-T5 GLUE LoRA r=16'], 'r_eff_norm', 'mean'):.4f}")
    print("  LoRA r=16 imposes a strict rank ≤ 16 structure on each Δ — peak_ratio blows up")
    print("  because past the LoRA subspace eigenvalues are bona-fide zero (numerical noise only).")


if __name__ == "__main__":
    main()
