"""Aggregate lm_eval results for the Llama-3.2-3B hyperparameter sweep.

Reads JSON reports under
``outputs/llama32_3b_mergebench/sweep/eval_results/<config>/<task>/.../results_*.json``
and emits:

* ``outputs/yongxianwei_merging/summary/llama32_3b_tuned.csv``
* ``outputs/yongxianwei_merging/summary/llama32_3b_tuned.md``

Configs are grouped by family and sorted by avg descending within each family.
Also includes the 5 defaults (iwudi, swudi, aswudi, swudi_align, wudi) for
direct comparison.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_EVAL_DIR = REPO_ROOT / "outputs" / "llama32_3b_mergebench" / "sweep" / "eval_results"
DEFAULT_EVAL_DIR = REPO_ROOT / "outputs" / "llama32_3b_mergebench" / "eval_results"
SUMMARY_DIR = REPO_ROOT / "outputs" / "yongxianwei_merging" / "summary"

LANGS = ["fr", "es", "de", "ru"]
ML_FAMILIES = {"m_mmlu", "arc", "hellaswag"}

TASKS = [
    "gsm8k_cot",
    "humaneval_plus",
    "mbpp_plus",
    "ifeval",
    "truthfulqa",
    "m_mmlu",
    "arc",
    "hellaswag",
]

# name -> (family, display label, hparam spec)
SWEEP_SPECS: Dict[str, Tuple[str, str, str]] = {
    "iwudi_t200":               ("IWUDI",       "IWUDI (t=200)",                       "exp_time=200"),
    "iwudi_t500":               ("IWUDI",       "IWUDI (t=500)",                       "exp_time=500"),
    "iwudi_t1000":              ("IWUDI",       "IWUDI (t=1000)",                      "exp_time=1000"),
    "swudi_t200_r090":          ("SWUDI",       "SWUDI (t=200, r=0.90)",               "t=200,r=0.90"),
    "swudi_t300_r085":          ("SWUDI",       "SWUDI (t=300, r=0.85)",               "t=300,r=0.85"),
    "swudi_t300_r095":          ("SWUDI",       "SWUDI (t=300, r=0.95)",               "t=300,r=0.95"),
    "swudi_t500_r090":          ("SWUDI",       "SWUDI (t=500, r=0.90)",               "t=500,r=0.90"),
    "swudi_t1000_r085":         ("SWUDI",       "SWUDI (t=1000, r=0.85)",              "t=1000,r=0.85"),
    "swudi_align_rho03":        ("SWUDI+",      "SWUDI+ (t=1300, r=0.65, ρ=0.3)",      "t=1300,r=0.65,rho=0.3"),
    "swudi_align_rho07":        ("SWUDI+",      "SWUDI+ (t=1300, r=0.65, ρ=0.7)",      "t=1300,r=0.65,rho=0.7"),
    "aswudi_entropy":           ("ASWUDI",      "ASWUDI (entropy)",                    "rule=entropy"),
    "aswudi_gavish_donoho":     ("ASWUDI",      "ASWUDI (gavish_donoho)",              "rule=gavish_donoho"),
    "aswudi_d_aware_entropy":   ("ASWUDI",      "ASWUDI (d_aware_entropy)",            "rule=d_aware_entropy"),
}

# CLIP-default configs already in outputs/llama32_3b_mergebench/eval_results/
DEFAULT_SPECS: Dict[str, Tuple[str, str, str]] = {
    "iwudi":        ("IWUDI",  "IWUDI (t=300, CLIP default)",                 "exp_time=300"),
    "swudi":        ("SWUDI",  "SWUDI (t=1300, r=0.65, CLIP default)",        "t=1300,r=0.65"),
    "swudi_align":  ("SWUDI+", "SWUDI+ (t=1300, r=0.65, ρ=0.5, CLIP default)", "t=1300,r=0.65,rho=0.5"),
    "aswudi":       ("ASWUDI", "ASWUDI (participation_sqrt, default)",        "rule=participation_sqrt"),
    "wudi":         ("WUDI",   "WUDI (iter=300, lr=1e-5, default)",           "iter=300"),
}

BASELINE_SPECS: Dict[str, Tuple[str, str, str]] = {
    "task_arithmetic": ("BASE", "task_arithmetic (scaling=0.3)", "baseline"),
    "simple_average":  ("BASE", "simple_average",                "baseline"),
}


def _latest_json(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name)[-1]


def _score(task: str, task_results: Dict[str, Dict]) -> Optional[float]:
    if task == "gsm8k_cot":
        return task_results.get("gsm8k_cot", {}).get("exact_match,flexible-extract")
    if task == "humaneval_plus":
        r = task_results.get("humaneval_plus", {})
        for key in ("pass@1,create_test", "pass@1,none", "pass_at_1,create_test", "pass_at_1,none"):
            if key in r:
                return r[key]
        return None
    if task == "mbpp_plus":
        r = task_results.get("mbpp_plus", {})
        for key in ("pass_at_1,none", "pass@1,none", "pass_at_1,create_test"):
            if key in r:
                return r[key]
        return None
    if task == "ifeval":
        r = task_results.get("ifeval", {})
        return r.get("prompt_level_strict_acc,none") or r.get("prompt_level_strict_acc")
    if task == "truthfulqa":
        r = task_results.get("truthfulqa_mc2") or task_results.get("truthfulqa")
        if r is None:
            return None
        return r.get("acc,none") or r.get("acc")
    r = task_results.get(task, {})
    return r.get("acc,none") or r.get("acc")


def _load_one(base_dir: Path, config: str, task_name: str) -> Optional[float]:
    """Load the scalar for a single concrete task (e.g. ``m_mmlu_es``)."""
    task_dir = base_dir / config / task_name
    if not task_dir.exists():
        return None
    jsons = list(task_dir.glob("*/results_*.json"))
    json_path = _latest_json(jsons)
    if json_path is None:
        return None
    with json_path.open() as fh:
        payload = json.load(fh)
    return _score(task_name, payload.get("results", {}))


def _load_scores(base_dir: Path, config: str) -> Dict[str, Optional[float]]:
    """8-column compressed scores: family columns are mean over available langs."""
    scores: Dict[str, Optional[float]] = {}
    for task in TASKS:
        if task in ML_FAMILIES:
            per_lang = []
            for lang in LANGS:
                v = _load_one(base_dir, config, f"{task}_{lang}")
                if v is not None:
                    per_lang.append(v)
            scores[task] = sum(per_lang) / len(per_lang) if per_lang else None
        else:
            scores[task] = _load_one(base_dir, config, task)
    return scores


def _rows_for(base_dir: Path, specs: Dict[str, Tuple[str, str, str]]):
    rows = []
    for config, (family, label, hparams) in specs.items():
        scores = _load_scores(base_dir, config)
        valid = [v for v in scores.values() if v is not None]
        avg = sum(valid) / len(valid) if len(valid) == len(TASKS) else (
            sum(valid) / len(valid) if valid else float("nan")
        )
        rows.append({
            "family": family,
            "config": config,
            "label": label,
            "hparams": hparams,
            "n_complete": len(valid),
            **scores,
            "avg": avg,
        })
    return rows


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_rows += _rows_for(SWEEP_EVAL_DIR, SWEEP_SPECS)
    all_rows += _rows_for(DEFAULT_EVAL_DIR, DEFAULT_SPECS)
    all_rows += _rows_for(DEFAULT_EVAL_DIR, BASELINE_SPECS)

    # sort by (family, -avg) so sweeps within a family are grouped
    family_order = {"IWUDI": 0, "SWUDI": 1, "SWUDI+": 2, "ASWUDI": 3, "WUDI": 4, "BASE": 5}
    all_rows.sort(key=lambda r: (family_order.get(r["family"], 99),
                                  -(r["avg"] if r["avg"] == r["avg"] else -1)))

    csv_path = SUMMARY_DIR / "llama32_3b_tuned.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["family", "config", "label", "hparams", "n_complete", *TASKS, "avg"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    k: ("" if v is None else (f"{v:.4f}" if isinstance(v, float) else v))
                    for k, v in row.items()
                }
            )

    md_path = SUMMARY_DIR / "llama32_3b_tuned.md"
    with md_path.open("w") as fh:
        fh.write("# Llama-3.2-3B MergeBench — hyperparameter sweep (tuned vs defaults)\n\n")
        fh.write(
            "seed=42; modelpool=CausalLMPool/mergebench/Llama-3.2-3B; "
            "8 lm_eval tasks (lm_eval 0.4.11 + transformers 5.8.0). "
            "13 new configs ran with `exclude_param_names_regex=[embed_tokens,lm_head]`. "
            "All defaults are previously tuned on CLIP-ViT-B/32 TA8; this sweep searches "
            "for Llama-native hyperparameters.\n\n"
        )
        header = "| family | config / hparams | " + " | ".join(TASKS) + " | avg |\n"
        sep = "|---|---|" + "|".join(["---:"] * (len(TASKS) + 1)) + "|\n"
        fh.write(header)
        fh.write(sep)
        for row in all_rows:
            vals = " | ".join(
                ("—" if row[t] is None else f"{row[t]:.4f}") for t in TASKS
            )
            avg = row["avg"]
            marker = ""
            if row["n_complete"] < len(TASKS):
                marker = f" (partial {row['n_complete']}/{len(TASKS)})"
            avg_str = f"**{avg:.4f}**{marker}" if avg == avg else "—"
            fh.write(f"| {row['family']} | `{row['label']}` | {vals} | {avg_str} |\n")
        fh.write("\n")
        fh.write(
            "metrics: gsm8k_cot = exact_match (flexible); "
            "humaneval_plus / mbpp_plus = pass@1; "
            "ifeval = prompt_level_strict_acc; "
            "truthfulqa = MC2 acc; "
            "m_mmlu / arc / hellaswag = acc, mean over the languages evaluated for "
            "each method (fr-only legacy methods collapse to French; methods with all "
            "4 langs report fr/es/de/ru means). "
            "avg = unweighted mean of 8 tasks (partial rows averaged over available tasks).\n"
        )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for row in all_rows:
        vals = ", ".join(
            f"{t}={row[t]:.4f}" if row[t] is not None else f"{t}=?" for t in TASKS
        )
        avg = row["avg"]
        avg_str = f"{avg:.4f}" if avg == avg else "nan"
        print(f"  [{row['family']:<6}] {row['label']:<50} avg={avg_str}  n={row['n_complete']}/8")


if __name__ == "__main__":
    main()