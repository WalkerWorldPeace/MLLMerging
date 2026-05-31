"""Aggregate lm_eval results for Llama-3.2-3B MergeBench merged models.

Reads every JSON report under
``outputs/llama32_3b_mergebench/eval_results/<method>/<task>/.../results_*.json``,
picks the canonical scalar score per task, and emits:

* ``outputs/yongxianwei_merging/summary/llama32_3b_mergebench.csv``
* ``outputs/yongxianwei_merging/summary/llama32_3b_mergebench.md``
* ``outputs/yongxianwei_merging/summary/llama32_3b_mergebench_lang.csv``
* ``outputs/yongxianwei_merging/summary/llama32_3b_mergebench_lang.md``

Methods are sorted by average score descending.

Task -> primary metric:

* gsm8k_cot      -> ``exact_match,flexible-extract``
* humaneval_plus -> ``pass@1,create_test``   (lm_eval 0.4.11 code_eval)
* mbpp_plus      -> ``pass_at_1,none``       (lm_eval 0.4.11 code_eval)
* ifeval         -> ``prompt_level_strict_acc``
* truthfulqa     -> ``acc``                  (taken from the ``truthfulqa_mc2`` subtask)
* m_mmlu / arc / hellaswag -> mean over languages in LANGS that have results

For the multilingual family columns (``m_mmlu``, ``arc``, ``hellaswag``), the
score is the unweighted mean over LANGS=("fr","es","de","ru") of the languages
that have results on disk (legacy fr-only methods report fr alone, so historical
numbers are bit-identical; methods evaluated on all 4 languages report 4-lang
means). The companion ``*_lang.{md,csv}`` file shows per-language detail.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "outputs" / "llama32_3b_mergebench" / "eval_results"
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

# Per-language detail columns (5 base + 12 multilingual = 17 columns).
LANG_TASKS = [
    "gsm8k_cot",
    "humaneval_plus",
    "mbpp_plus",
    "ifeval",
    "truthfulqa",
] + [f"{fam}_{lang}" for fam in ("m_mmlu", "arc", "hellaswag") for lang in LANGS]


def _latest_json(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name)[-1]


def _score(task: str, task_results: Dict[str, Dict]) -> Optional[float]:
    """Pick the canonical scalar for `task` from the `results` dict.

    `task` here is always a concrete lm_eval task name (e.g. ``m_mmlu_fr``,
    ``arc_de``). Family aggregation happens in ``_load_*_scores``.
    """
    if task == "gsm8k_cot":
        r = task_results.get("gsm8k_cot", {})
        return r.get("exact_match,flexible-extract")
    if task == "humaneval_plus":
        r = task_results.get("humaneval_plus", {})
        # lm_eval 0.4.11 exposes pass@1 under the 'create_test' filter
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
        # prompt_level_strict_acc is the most commonly reported ifeval number
        v = r.get("prompt_level_strict_acc,none")
        if v is None:
            v = r.get("prompt_level_strict_acc")
        return v
    if task == "truthfulqa":
        # truthfulqa is a group; lm_eval fills subtask scores
        r = task_results.get("truthfulqa_mc2") or task_results.get("truthfulqa")
        if r is None:
            return None
        for key in ("acc,none", "acc"):
            if key in r:
                return r[key]
        return None
    # acc for simple MC tasks (incl. m_mmlu_*, arc_*, hellaswag_*)
    r = task_results.get(task, {})
    for key in ("acc,none", "acc"):
        if key in r:
            return r[key]
    return None


def _load_one(method: str, task_name: str) -> Optional[float]:
    """Load the scalar for a single concrete task (e.g. ``m_mmlu_es``)."""
    task_dir = EVAL_DIR / method / task_name
    if not task_dir.exists():
        return None
    jsons = list(task_dir.glob("*/results_*.json"))
    json_path = _latest_json(jsons)
    if json_path is None:
        return None
    with json_path.open() as fh:
        payload = json.load(fh)
    return _score(task_name, payload.get("results", {}))


def _load_scores(method: str) -> Dict[str, Optional[float]]:
    """8-column compressed scores: family columns are mean over available langs."""
    scores: Dict[str, Optional[float]] = {}
    for task in TASKS:
        if task in ML_FAMILIES:
            per_lang = []
            for lang in LANGS:
                v = _load_one(method, f"{task}_{lang}")
                if v is not None:
                    per_lang.append(v)
            scores[task] = sum(per_lang) / len(per_lang) if per_lang else None
        else:
            scores[task] = _load_one(method, task)
    return scores


def _load_lang_scores(method: str) -> Dict[str, Optional[float]]:
    """17-column per-language scores: every (family, lang) is its own column."""
    return {task: _load_one(method, task) for task in LANG_TASKS}


def main() -> None:
    if not EVAL_DIR.exists():
        raise SystemExit(f"No eval dir at {EVAL_DIR}")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    lang_rows = []
    for method_dir in sorted(EVAL_DIR.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        scores = _load_scores(method)
        valid = [v for v in scores.values() if v is not None]
        avg = sum(valid) / len(valid) if valid else float("nan")
        rows.append({"method": method, **scores, "avg": avg})
        lang_rows.append({"method": method, **_load_lang_scores(method)})

    rows.sort(key=lambda r: (-r["avg"] if r["avg"] == r["avg"] else 1))
    # Match main row ordering for the lang detail file.
    method_order = {r["method"]: i for i, r in enumerate(rows)}
    lang_rows.sort(key=lambda r: method_order[r["method"]])

    csv_path = SUMMARY_DIR / "llama32_3b_mergebench.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", *TASKS, "avg"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: ("" if v is None else (f"{v:.4f}" if isinstance(v, float) else v))
                    for k, v in row.items()
                }
            )

    md_path = SUMMARY_DIR / "llama32_3b_mergebench.md"
    with md_path.open("w") as fh:
        fh.write("# yongxianwei_merging on Llama-3.2-3B MergeBench (5 experts)\n\n")
        fh.write(
            "seed=42; CausalLMPool/mergebench/Llama-3.2-3B with "
            "`merge_backbone=true` / `exclude_param_names_regex=[embed_tokens,lm_head]` "
            "(lm_head/embed shapes differ between base 128256 and experts 128320).\n"
            "Evaluation via lm_eval 0.4.11 + transformers 5.8.0, 8 tasks, "
            "batch_size=8, bfloat16, single H20 per method (7-way parallel).\n\n"
            "Multilingual family columns (`m_mmlu`, `arc`, `hellaswag`) are unweighted "
            "means over the languages evaluated for each method (fr-only legacy methods "
            "collapse to French; methods evaluated on all 4 languages report fr/es/de/ru "
            "means). Per-language detail in `llama32_3b_mergebench_lang.md`.\n\n"
        )
        header = "| method | " + " | ".join(TASKS) + " | avg |\n"
        sep = "|---|" + "|".join(["---:"] * (len(TASKS) + 1)) + "|\n"
        fh.write(header)
        fh.write(sep)
        for row in rows:
            vals = " | ".join(
                ("—" if row[t] is None else f"{row[t]:.4f}") for t in TASKS
            )
            avg = row["avg"]
            avg_str = f"**{avg:.4f}**" if avg == avg else "—"
            fh.write(f"| `{row['method']}` | {vals} | {avg_str} |\n")
        fh.write("\n")
        fh.write(
            "metrics:\n"
            "  - `gsm8k_cot`: exact_match (flexible-extract) — generative math\n"
            "  - `humaneval_plus`: pass@1 (create_test filter) — code generation\n"
            "  - `mbpp_plus`: pass@1 — code generation\n"
            "  - `ifeval`: prompt_level_strict_acc — instruction following\n"
            "  - `truthfulqa`: acc (MC2 subtask) — safety/truthfulness\n"
            "  - `m_mmlu`, `arc`, `hellaswag`: acc — multilingual MC, mean over the "
            "languages evaluated for each method (fr-only legacy methods collapse to "
            "French; swudi_t300_r085 is fr/es/de/ru). Per-language detail in "
            "`llama32_3b_mergebench_lang.md`.\n"
            "  - avg: unweighted mean across 8 tasks\n"
        )

    # Per-language detail file (17 columns).
    lang_csv_path = SUMMARY_DIR / "llama32_3b_mergebench_lang.csv"
    with lang_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", *LANG_TASKS])
        writer.writeheader()
        for row in lang_rows:
            writer.writerow(
                {
                    k: ("" if v is None else (f"{v:.4f}" if isinstance(v, float) else v))
                    for k, v in row.items()
                }
            )

    lang_md_path = SUMMARY_DIR / "llama32_3b_mergebench_lang.md"
    with lang_md_path.open("w") as fh:
        fh.write("# Llama-3.2-3B MergeBench — per-language detail\n\n")
        fh.write(
            "Companion to `llama32_3b_mergebench.md`. Shows the 12 individual "
            "multilingual subtask scores (fr/es/de/ru × m_mmlu/arc/hellaswag) plus "
            "the 5 base tasks. Methods evaluated only on French show `—` for es/de/ru.\n\n"
        )
        header = "| method | " + " | ".join(LANG_TASKS) + " |\n"
        sep = "|---|" + "|".join(["---:"] * len(LANG_TASKS)) + "|\n"
        fh.write(header)
        fh.write(sep)
        for row in lang_rows:
            vals = " | ".join(
                ("—" if row[t] is None else f"{row[t]:.4f}") for t in LANG_TASKS
            )
            fh.write(f"| `{row['method']}` | {vals} |\n")
        fh.write("\n")
        fh.write(
            "metrics: identical to the compressed table; family-mean rows live in "
            "`llama32_3b_mergebench.md`.\n"
        )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {lang_csv_path}")
    print(f"wrote {lang_md_path}")
    for row in rows:
        vals = ", ".join(
            f"{t}={row[t]:.4f}" if row[t] is not None else f"{t}=?" for t in TASKS
        )
        print(f"  {row['method']:<32} avg={row['avg']:.4f}  ({vals})")


if __name__ == "__main__":
    main()
