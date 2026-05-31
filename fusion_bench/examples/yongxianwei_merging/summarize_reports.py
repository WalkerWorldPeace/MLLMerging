"""Summarize yongxianwei_merging evaluation reports into CSV + Markdown.

Scans ``outputs/yongxianwei_merging/reports/clip-vit-base-patch32_TA8/*.json``
and ``outputs/yongxianwei_merging/baselines/*.json`` and produces
``outputs/yongxianwei_merging/summary/clip-vit-base-patch32_TA8.csv`` and
``.md`` sorted by average accuracy (descending).

Each report is the output of FusionBench ``CLIPVisionModelTaskPool``; it
typically nests per-task metrics by task name with an ``average`` entry.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


TA8_TASKS = [
    "sun397",
    "stanford-cars",
    "resisc45",
    "eurosat",
    "svhn",
    "gtsrb",
    "mnist",
    "dtd",
]

# Reference numbers for CLIP-ViT-B/32 TA8 commonly cited in the docs and
# literature. They are reproduced here for side-by-side comparison; refer
# to https://tanganke.github.io/fusion_bench/ for the original sources.
OFFICIAL_REFERENCE: Dict[str, float] = {
    "pretrained": 48.2,
    "task_arithmetic": 70.1,
    "ties_merging": 72.4,
    "regmean": 71.8,
    "adamerging": 80.1,
    "opcm": 74.6,
}


def _first_float(obj, keys):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            val = obj[key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _extract_row(name: str, report_path: Path, reference: Optional[float]) -> Dict:
    with report_path.open() as f:
        data = json.load(f)

    # report structure: {"taskpool": {...}, ...} or top-level per task.
    taskpool = data.get("taskpool") if isinstance(data, dict) else None
    if isinstance(taskpool, dict):
        data = taskpool

    row = {
        "method": name,
        "report_path": str(report_path),
        "official_reference": reference,
    }

    # Per-task accuracies
    for task in TA8_TASKS:
        task_entry = data.get(task) if isinstance(data, dict) else None
        acc = None
        if isinstance(task_entry, dict):
            acc = _first_float(task_entry, ("accuracy", "acc", "top1"))
        row[f"{task}_acc"] = acc

    average_entry = data.get("average") if isinstance(data, dict) else None
    if isinstance(average_entry, dict):
        row["average_accuracy"] = _first_float(
            average_entry, ("accuracy", "acc", "top1")
        )
        row["average_loss"] = _first_float(average_entry, ("loss",))
    else:
        accs = [row[f"{t}_acc"] for t in TA8_TASKS if row[f"{t}_acc"] is not None]
        row["average_accuracy"] = sum(accs) / len(accs) if accs else None
        row["average_loss"] = None

    return row


def _collect(directory: Path, reference_map: Dict[str, float]) -> List[Dict]:
    if not directory.exists():
        return []
    rows = []
    for report_path in sorted(directory.glob("*.json")):
        name = report_path.stem
        rows.append(_extract_row(name, report_path, reference_map.get(name)))
    return rows


def _write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "method",
        "average_accuracy",
        "average_loss",
    ] + [f"{t}_acc" for t in TA8_TASKS] + ["official_reference", "report_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_md(rows: List[Dict], out_path: Path) -> None:
    headers = [
        "method",
        "average_accuracy",
        "average_loss",
        *[f"{t}_acc" for t in TA8_TASKS],
        "official_reference",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            cells = []
            for h in headers:
                v = row.get(h)
                if v is None:
                    cells.append("")
                elif isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        default="outputs/yongxianwei_merging/reports/clip-vit-base-patch32_TA8",
        help="Directory containing evaluation reports for ported methods.",
    )
    parser.add_argument(
        "--baselines",
        default="outputs/yongxianwei_merging/baselines",
        help="Directory containing baseline reports (dummy, simple_average, task_arithmetic, ...).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/yongxianwei_merging/summary",
        help="Output directory for summary.csv and summary.md",
    )
    parser.add_argument(
        "--benchmark",
        default="clip-vit-base-patch32_TA8",
        help="Benchmark name used in the output filenames.",
    )
    args = parser.parse_args()

    rows = []
    rows.extend(_collect(Path(args.reports), OFFICIAL_REFERENCE))
    rows.extend(_collect(Path(args.baselines), OFFICIAL_REFERENCE))

    if not rows:
        print("No reports found.")
        return

    rows.sort(key=lambda r: (r.get("average_accuracy") or 0.0), reverse=True)

    out_dir = Path(args.out_dir)
    _write_csv(rows, out_dir / f"{args.benchmark}.csv")
    _write_md(rows, out_dir / f"{args.benchmark}.md")
    print(f"Wrote {len(rows)} rows to {out_dir}/{args.benchmark}.{{csv,md}}")


if __name__ == "__main__":
    main()
