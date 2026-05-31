"""Aggregate Flan-T5-base GLUE (LoRA r=16) reports into a markdown + CSV table.

Reads every ``outputs/yongxianwei_merging/reports/flan-t5-base_glue_lora16/<method>.json``
produced by the GLUE fusion_bench runs, extracts per-task accuracy / spearman_rho,
computes the average across the 8 GLUE tasks (treating ``spearman_rho`` for stsb
and ``accuracy`` for the other seven as equal-weight metrics), and writes:

* ``outputs/yongxianwei_merging/summary/flan-t5-base_glue_lora16.csv``
* ``outputs/yongxianwei_merging/summary/flan-t5-base_glue_lora16.md``

Methods are sorted by average score descending.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = (
    REPO_ROOT / "outputs" / "yongxianwei_merging" / "reports" / "flan-t5-base_glue_lora16"
)
SUMMARY_DIR = REPO_ROOT / "outputs" / "yongxianwei_merging" / "summary"
TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]


def _score(task_result: Dict) -> float:
    if "accuracy" in task_result:
        return float(task_result["accuracy"])
    if "spearman_rho" in task_result:
        return float(task_result["spearman_rho"])
    raise KeyError(f"no accuracy/spearman_rho in {task_result!r}")


def main() -> None:
    if not REPORTS_DIR.exists():
        raise SystemExit(f"No reports dir at {REPORTS_DIR}")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for json_path in sorted(REPORTS_DIR.glob("*.json")):
        method = json_path.stem
        with json_path.open() as fh:
            report = json.load(fh)
        try:
            scores = [_score(report[t]) for t in TASKS]
        except KeyError as exc:
            print(f"[skip] {method}: {exc}")
            continue
        avg = sum(scores) / len(scores)
        rows.append({"method": method, **dict(zip(TASKS, scores)), "avg": avg})

    rows.sort(key=lambda r: r["avg"], reverse=True)

    csv_path = SUMMARY_DIR / "flan-t5-base_glue_lora16.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", *TASKS, "avg"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    md_path = SUMMARY_DIR / "flan-t5-base_glue_lora16.md"
    with md_path.open("w") as fh:
        fh.write("# yongxianwei_merging on Flan-T5-base GLUE (LoRA r=16)\n\n")
        fh.write("seed=42; Seq2SeqLMPool/flan-t5-base_glue_lora16 + flan-t5_glue_text_generation.\n\n")
        header = "| method | " + " | ".join(TASKS) + " | avg |\n"
        sep = "|---|" + "|".join(["---:"] * (len(TASKS) + 1)) + "|\n"
        fh.write(header)
        fh.write(sep)
        for row in rows:
            vals = " | ".join(f"{row[t]:.4f}" for t in TASKS)
            fh.write(f"| `{row['method']}` | {vals} | **{row['avg']:.4f}** |\n")
        fh.write("\n")
        fh.write("metric: `accuracy` for cola/mnli/mrpc/qnli/qqp/rte/sst2; `spearman_rho` for stsb.\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for row in rows:
        print(f"  {row['method']:<16} avg={row['avg']:.4f}")


if __name__ == "__main__":
    main()
