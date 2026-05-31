"""Summarise the 8 paper baseline re-evaluations on full-dataset MathVista_MINI
and MathVision_MINI Overall acc, then recompute the 10-task average (TODO §3.5).

Inputs:
  $VLMEVAL_WORKDIR (default outputs/.../mllm_full_math/vlmevalkit) — VLMEvalKit
    work directory containing per-model subdirs with
    ``*_MathVista_MINI_gpt-4o-mini_score.csv`` and
    ``*_MathVision_MINI_gpt-4o-mini_score.csv``.
  --legacy_metrics_csv (optional) — existing per-baseline 8-task metrics
    (VizWiz/GQA/ChartQA/TextVQA/OCRVQA/RefCOCO[+/g] + geom-only Math) so we can
    keep the geometry-subset numbers as a column for reviewer reference.

Outputs (under $EVAL_ROOT, default outputs/.../mllm_full_math/):
  qwen2vl_table3_full_math.md   — Qwen Table 3 with both geom and Overall Math.
  internvl_table2_full_math.md  — InternVL Table 2 likewise.
  full_math_raw_scores.csv      — long-form raw scores (model, dataset, acc, tot).
  full_math_recomputed_10avg.csv — wide table with both 10-avg flavours.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
    ("weight_average",  "Weight Average"),
    ("task_arithmetic", "Task Arithmetic"),
    ("ties",            "TIES Merging"),
    ("dare_ta",         "TA w/ DARE"),
    ("dare_ties",       "TIES w/ DARE"),
    ("svd",             "TSV(svd) Merging"),
    ("iso",             "Iso-C"),
    ("wudi",            "WUDI Merging"),
]

DATASETS = ("MathVista_MINI", "MathVision_MINI")

# 10-task ordering — see TODO §0.
TEN_TASK_ORDER = [
    "VizWiz", "GQA", "MathVista_MINI", "MathVision_MINI",
    "ChartQA", "TextVQA", "OCRVQA",
    "RefCOCO", "RefCOCO+", "RefCOCOg",
]


def read_overall_acc(score_csv: Path) -> tuple[float, int]:
    """Robust reader matching TODO §3.5 helper."""
    df = pd.read_csv(score_csv)
    key = "Task&Skill" if "Task&Skill" in df.columns else "Subject"
    row = df[df[key] == "Overall"]
    assert len(row) == 1, f"missing Overall row in {score_csv}"
    return float(row.iloc[0]["acc"]), int(row.iloc[0]["tot"])


def discover_scores(workdir: Path, family: str) -> dict:
    """Returns {(method_tag, dataset): (acc, tot)}."""
    out = {}
    for tag, _ in METHOD_ORDER:
        model_name = f"merge_{family}_{tag}_fullmath"
        sub = workdir / model_name
        for ds in DATASETS:
            f = sub / f"{model_name}_{ds}_gpt-4o-mini_score.csv"
            if f.exists():
                out[(tag, ds)] = read_overall_acc(f)
            else:
                out[(tag, ds)] = (float("nan"), 0)
    return out


def load_legacy_metrics(path: Path | None) -> dict:
    """Returns {(family, method_tag): {dataset: float}} for the 8 non-Math
    metrics + the geom-subset Math reference. Optional."""
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        family = str(row["family"])
        method = str(row["method_tag"])
        out.setdefault((family, method), {})
        for ds in TEN_TASK_ORDER:
            out[(family, method)][ds] = (
                float(row[ds]) if ds in df.columns and pd.notna(row.get(ds)) else None
            )
        # geometry-subset Math (legacy paper numbers)
        for col in ("MathVista_MINI_geom", "MathVision_MINI_geom"):
            if col in df.columns and pd.notna(row.get(col)):
                out[(family, method)][col] = float(row[col])
    return out


def fmt(x):
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "—"
    return f"{x:.2f}"


def avg(vals):
    """Mean of values that are not None / NaN."""
    keep = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
    return sum(keep) / len(keep) if keep else None


def write_family_table(out_md: Path, family: str, paper_table_label: str,
                       full_scores: dict, legacy: dict, raw_rows: list):
    rows = []
    for tag, paper_name in METHOD_ORDER:
        legacy_metrics = legacy.get((family, tag), {})
        mathvista_full = full_scores.get((tag, "MathVista_MINI"), (float("nan"), 0))[0]
        mathvision_full = full_scores.get((tag, "MathVision_MINI"), (float("nan"), 0))[0]
        mathvista_geom = legacy_metrics.get("MathVista_MINI_geom")
        mathvision_geom = legacy_metrics.get("MathVision_MINI_geom")

        non_math = [legacy_metrics.get(d) for d in TEN_TASK_ORDER
                    if d not in ("MathVista_MINI", "MathVision_MINI")]
        ten_full = avg(non_math + [mathvista_full, mathvision_full])
        ten_geom = avg(non_math + [mathvista_geom, mathvision_geom])

        rows.append({
            "method":             paper_name,
            "method_tag":         tag,
            "MathVista_geom":     mathvista_geom,
            "MathVista_Overall":  mathvista_full,
            "MathVision_geom":    mathvision_geom,
            "MathVision_Overall": mathvision_full,
            "10-avg (geom math)": ten_geom,
            "10-avg (full math)": ten_full,
            **{d: legacy_metrics.get(d) for d in TEN_TASK_ORDER
               if d not in ("MathVista_MINI", "MathVision_MINI")},
        })

        for ds, (acc, tot) in full_scores.items():
            if ds[0] == tag:
                raw_rows.append((family, tag, ds[1], acc, tot))

    cols = (["method", "method_tag", "MathVista_geom", "MathVista_Overall",
             "MathVision_geom", "MathVision_Overall"]
            + [d for d in TEN_TASK_ORDER if d not in ("MathVista_MINI", "MathVision_MINI")]
            + ["10-avg (geom math)", "10-avg (full math)"])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write(f"# {paper_table_label} — full-dataset MathVista/MathVision\n\n")
        f.write("Math columns:\n")
        f.write("- `*_geom` = geometry-subset mean (paper-original; reviewer ablation reference).\n")
        f.write("- `*_Overall` = full-dataset Overall acc from `_score.csv`.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) if c not in ("method", "method_tag")
                                      else str(r.get(c, "")) for c in cols) + " |\n")
        f.write("\nSource: per-method `merge_<family>_<tag>_fullmath_*_score.csv` "
                "rows where `Task&Skill==Overall` / `Subject==Overall`.\n")


def main():
    p = argparse.ArgumentParser()
    _repo_root = Path(__file__).resolve().parents[2]
    _default_eval_root = _repo_root / "outputs" / "yongxianwei_merging" / "mllm_full_math"
    p.add_argument("--workdir", default=os.environ.get(
        "VLMEVAL_WORKDIR",
        str(_default_eval_root / "vlmevalkit"),
    ))
    p.add_argument("--eval_root", default=os.environ.get(
        "EVAL_ROOT",
        str(_default_eval_root),
    ))
    p.add_argument("--legacy_metrics_csv", default=None,
                   help="CSV with per-baseline 8-task metrics + geom Math; "
                        "expected columns: family, method_tag, "
                        + ", ".join(TEN_TASK_ORDER) + ", "
                        "MathVista_MINI_geom, MathVision_MINI_geom.")
    args = p.parse_args()

    workdir = Path(args.workdir)
    eval_root = Path(args.eval_root)
    legacy = load_legacy_metrics(Path(args.legacy_metrics_csv)
                                  if args.legacy_metrics_csv else None)

    qwen_scores = discover_scores(workdir, "qwen")
    internvl_scores = discover_scores(workdir, "internvl")

    raw_rows = []
    write_family_table(eval_root / "qwen2vl_table3_full_math.md",
                       "qwen", "Qwen2-VL-7B Table 3",
                       qwen_scores, legacy, raw_rows)
    write_family_table(eval_root / "internvl_table2_full_math.md",
                       "internvl", "InternVL2_5-1B Table 2",
                       internvl_scores, legacy, raw_rows)

    # raw long-form CSV
    with (eval_root / "full_math_raw_scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "method_tag", "dataset", "Overall_acc", "tot"])
        for row in raw_rows:
            w.writerow(row)

    # 10-avg wide CSV
    with (eval_root / "full_math_recomputed_10avg.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["family", "method_tag",
                "MathVista_geom", "MathVista_Overall",
                "MathVision_geom", "MathVision_Overall",
                "10-avg (geom math)", "10-avg (full math)",
                "delta_full_minus_geom"]
        w.writerow(cols)
        for family, scores in (("qwen", qwen_scores), ("internvl", internvl_scores)):
            for tag, _ in METHOD_ORDER:
                lm = legacy.get((family, tag), {})
                mathvista_full = scores.get((tag, "MathVista_MINI"), (float("nan"), 0))[0]
                mathvision_full = scores.get((tag, "MathVision_MINI"), (float("nan"), 0))[0]
                mathvista_geom = lm.get("MathVista_MINI_geom")
                mathvision_geom = lm.get("MathVision_MINI_geom")
                non_math = [lm.get(d) for d in TEN_TASK_ORDER
                            if d not in ("MathVista_MINI", "MathVision_MINI")]
                ten_full = avg(non_math + [mathvista_full, mathvision_full])
                ten_geom = avg(non_math + [mathvista_geom, mathvision_geom])
                delta = (ten_full - ten_geom) if (ten_full is not None and ten_geom is not None) else None
                w.writerow([family, tag,
                            fmt(mathvista_geom), fmt(mathvista_full),
                            fmt(mathvision_geom), fmt(mathvision_full),
                            fmt(ten_geom), fmt(ten_full), fmt(delta)])

    print(f"Wrote: {eval_root}/qwen2vl_table3_full_math.md")
    print(f"Wrote: {eval_root}/internvl_table2_full_math.md")
    print(f"Wrote: {eval_root}/full_math_raw_scores.csv")
    print(f"Wrote: {eval_root}/full_math_recomputed_10avg.csv")


if __name__ == "__main__":
    main()
