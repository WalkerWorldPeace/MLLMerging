"""CLI runner for the 8 paper baselines on InternVL2_5-1B (TODO §2.3).

Loads base + 5 experts ONCE and runs every requested method back-to-back; see
``run_merge_baseline.py`` for the same pattern on Qwen2-VL.

Methods supported: weight_average, task_arithmetic, ties, dare_ta, dare_ties,
svd, iso, wudi.

Examples:
    python run_merge_baseline_internvl.py \\
        --output_root /path/to/output/internvl2_5_1b
        # default --methods runs all 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from baseline_merging import (DEFAULT_HPARAMS, VALID_METHODS, merge_baseline)


DEFAULT_BASE = "OpenGVLab/InternVL2_5-1B"
DEFAULT_EXPERTS = [
    "yongxianwei/InternVL2_5-1B_OCR",
    "yongxianwei/InternVL2_5-1B_VQA",
    "yongxianwei/InternVL2_5-1B_Geometry",
    "yongxianwei/InternVL2_5-1B_Chart",
    "yongxianwei/InternVL2_5-1B_Grounding",
]
DEFAULT_EXCLUDE_REGEX = [
    "vision_model.*",
    ".*lm_head.*",
    ".*norm.*",
    ".*embed_tokens.*",
    ".*bias.*",
]
INTERNVL_REMOTE_CODE_FILES = [
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "conversation.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "preprocessor_config.json",
    "configuration.json",
]
TAG_PREFIX = "internvl"
PAPER_TABLE = "Table 2"
MODEL_FAMILY = "internvl2_5_1b"
PAPER_NAME_MAP = {
    "weight_average":  "Weight Average",
    "task_arithmetic": "Task Arithmetic",
    "ties":            "TIES Merging",
    "dare_ta":         "TA w/ DARE",
    "dare_ties":       "TIES w/ DARE",
    "svd":             "TSV(svd) Merging",
    "iso":             "Iso-C",
    "wudi":            "WUDI Merging",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(VALID_METHODS),
                   choices=list(VALID_METHODS))
    p.add_argument("--output_root", required=True)
    p.add_argument("--scaling_coefficient", type=float, default=None)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--experts", nargs="+", default=DEFAULT_EXPERTS)
    p.add_argument("--torch_dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dare_weight_mask_rate", type=float, default=None)
    p.add_argument("--ties_param_value_mask_rate", type=float, default=None)
    p.add_argument("--wudi_iter_num", type=int, default=None)
    p.add_argument("--wudi_lr", type=float, default=None)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def resolve_dtype(name: str):
    return {"float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def merged_kwargs(args, method):
    base = dict(DEFAULT_HPARAMS[method])
    if args.scaling_coefficient is not None:
        base["scaling_coefficient"] = float(args.scaling_coefficient)
    if args.dare_weight_mask_rate is not None and "dare_weight_mask_rate" in base:
        base["dare_weight_mask_rate"] = float(args.dare_weight_mask_rate)
    if args.ties_param_value_mask_rate is not None and "ties_param_value_mask_rate" in base:
        base["ties_param_value_mask_rate"] = float(args.ties_param_value_mask_rate)
    if args.wudi_iter_num is not None and "wudi_iter_num" in base:
        base["wudi_iter_num"] = int(args.wudi_iter_num)
    if args.wudi_lr is not None and "wudi_lr" in base:
        base["wudi_lr"] = float(args.wudi_lr)
    return base


def cache_states(models):
    out = []
    for m in models:
        out.append({n: t.detach().clone() for n, t in m.state_dict().items()})
    return out


def restore_states(models, caches):
    for m, c in zip(models, caches):
        m.load_state_dict(c, strict=True)


def copy_internvl_remote_code(output_path: str, base_repo: str):
    from huggingface_hub import snapshot_download
    snap = snapshot_download(base_repo, allow_patterns=INTERNVL_REMOTE_CODE_FILES)
    for f in INTERNVL_REMOTE_CODE_FILES:
        src = os.path.join(snap, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, f))


def write_manifest(output_path: str, args, method, kwargs, elapsed_sec):
    out = Path(output_path)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
        ).decode().strip()
    except Exception:
        commit = ""
    manifest = {
        "model_family":               MODEL_FAMILY,
        "paper_table":                PAPER_TABLE,
        "method":                     PAPER_NAME_MAP[method],
        "method_tag":                 method,
        "base_model":                 args.base,
        "expert_models":              args.experts,
        "exclude_param_names_regex":  DEFAULT_EXCLUDE_REGEX,
        "scaling_coefficient":        kwargs.get("scaling_coefficient"),
        "ties_param_value_mask_rate": kwargs.get("ties_param_value_mask_rate"),
        "dare_weight_mask_rate":      kwargs.get("dare_weight_mask_rate"),
        "wudi_iter_num":              kwargs.get("wudi_iter_num"),
        "wudi_lr":                    kwargs.get("wudi_lr"),
        "seed":                       args.seed if method.startswith("dare") else None,
        "torch_dtype":                args.torch_dtype,
        "torch_version":              torch.__version__,
        "transformers_version":       transformers.__version__,
        "source_commit":              commit,
        "merge_command":              " ".join(sys.argv),
        "checkpoint_path":            str(out.resolve()),
        "elapsed_sec":                round(elapsed_sec, 1),
    }
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))


def run_one(method, args, base_model, experts, base_cache, expert_caches, tokenizer):
    out_dir = os.path.join(args.output_root, f"{TAG_PREFIX}_{method}")
    if args.skip_existing and os.path.exists(os.path.join(out_dir, "merge_manifest.json")):
        print(f"[skip] {method} (manifest already at {out_dir})")
        return
    kwargs = merged_kwargs(args, method)
    print(f"\n========== merging method={method} kwargs={kwargs} ==========")
    t0 = time.time()
    print("[reset] restoring base + expert state_dicts ...")
    restore_states([base_model], [base_cache])
    restore_states(experts, expert_caches)

    merged = merge_baseline(
        method=method,
        base_model=base_model,
        models_to_merge=experts,
        exclude_param_names_regex=DEFAULT_EXCLUDE_REGEX,
        seed=args.seed,
        device=args.device,
        **kwargs,
    )

    print("[merge] applying merged params to base ...")
    state = base_model.state_dict()
    for k, v in merged.items():
        if k in state:
            state[k] = v.to(state[k].dtype) if v.dtype != state[k].dtype else v
    base_model.load_state_dict(state)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[save] {out_dir}")
    base_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    copy_internvl_remote_code(out_dir, args.base)
    elapsed = time.time() - t0
    write_manifest(out_dir, args, method, kwargs, elapsed)
    print(f"[done] {method} in {elapsed:.1f}s -> {out_dir}")
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    dtype = resolve_dtype(args.torch_dtype)

    print(f"[load] base   {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=True, use_fast=False,
    )
    base_model = AutoModel.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True,
    ).eval()

    experts = []
    for path in args.experts:
        print(f"[load] expert {path}")
        m = AutoModel.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True,
        ).eval()
        experts.append(m)

    print("[cache] snapshotting state_dicts for cross-method reset ...")
    base_cache = cache_states([base_model])[0]
    expert_caches = cache_states(experts)

    print(f"[plan] running methods: {args.methods}")
    for method in args.methods:
        try:
            run_one(method, args, base_model, experts,
                    base_cache, expert_caches, tokenizer)
        except Exception as e:
            print(f"[error] method={method} failed: {e}")
            raise

    print("\nAll done.")


if __name__ == "__main__":
    main()
