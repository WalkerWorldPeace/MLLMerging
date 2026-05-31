"""Wall-clock + peak-GPU-memory benchmark for SWUDI / ASWUDI / wudi2 on
Qwen2-VL-7B or InternVL2_5-1B multi-expert merging.

Loads base + 5 experts once, builds task vectors once, then for each
method runs a fresh merge with `torch.cuda.reset_peak_memory_stats` +
wall-clock timing. Skips checkpoint saving (we only want the merge cost).

Reuses the wudi_merging2 implementation from run_merge_wudi2*.py
(the LLaMA-Factory variant for Qwen2-VL, the InternVL variant for
InternVL2_5-1B), and swudi_merge / aswudi_merge from swudi_aswudi.py.

Usage:
    python benchmark_mllm_merge_cost.py --target qwen2vl  --gpu 1 \
        --output mllm_timing_qwen2vl.json
    python benchmark_mllm_merge_cost.py --target internvl --gpu 0 \
        --output mllm_timing_internvl.json
"""

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

# Locate run_merge_wudi2{,_internvl}.py and swudi_aswudi.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SWUDI_MOD = load_module("swudi_aswudi_local", HERE / "swudi_aswudi.py")
WUDI2_QWEN = load_module("wudi2_qwen_runner", HERE / "run_merge_wudi2.py")
WUDI2_INTERNVL = load_module("wudi2_internvl_runner", HERE / "run_merge_wudi2_internvl.py")


QWEN_BASE = "Qwen/Qwen2-VL-7B"
QWEN_EXPERTS = [
    "yongxianwei/Qwen2-VL-7B-OCR",
    "yongxianwei/Qwen2-VL-7B-VQA",
    "yongxianwei/Qwen2-VL-7B-Geometry",
    "yongxianwei/Qwen2-VL-7B-Chart",
    "yongxianwei/Qwen2-VL-7B-Grounding",
]
QWEN_EXCLUDE = ["visual..*", ".*embed_tokens.*", ".*lm_head.*", ".*norm.*", ".*bias.*"]

INTERNVL_BASE = "OpenGVLab/InternVL2_5-1B"
INTERNVL_EXPERTS = [
    "yongxianwei/InternVL2_5-1B_OCR",
    "yongxianwei/InternVL2_5-1B_VQA",
    "yongxianwei/InternVL2_5-1B_Geometry",
    "yongxianwei/InternVL2_5-1B_Chart",
    "yongxianwei/InternVL2_5-1B_Grounding",
]
INTERNVL_EXCLUDE = ["vision_model.*", ".*lm_head.*", ".*norm.*", ".*embed_tokens.*", ".*bias.*"]


def load_qwen():
    from transformers import Qwen2VLForConditionalGeneration
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_BASE, torch_dtype=torch.float16, trust_remote_code=True,
    ).eval()
    experts = [
        Qwen2VLForConditionalGeneration.from_pretrained(
            p, torch_dtype=torch.float16, trust_remote_code=True
        ).eval()
        for p in QWEN_EXPERTS
    ]
    return base, experts, QWEN_EXCLUDE


def load_internvl():
    from transformers import AutoModel
    base = AutoModel.from_pretrained(
        INTERNVL_BASE, torch_dtype=torch.float16, trust_remote_code=True,
    ).eval().cuda()
    experts = [
        AutoModel.from_pretrained(
            p, torch_dtype=torch.float16, trust_remote_code=True
        ).eval().cuda()
        for p in INTERNVL_EXPERTS
    ]
    return base, experts, INTERNVL_EXCLUDE


def time_method(name, fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated() / 1024**3
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{name}] elapsed={elapsed:.1f}s peak={peak:.2f}GiB baseline={baseline:.2f}GiB delta={peak-baseline:.2f}GiB", flush=True)
    del out
    torch.cuda.empty_cache()
    gc.collect()
    return {"elapsed_sec": elapsed, "peak_gib": peak,
            "baseline_gib": baseline, "delta_gib": peak - baseline}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["qwen2vl", "internvl"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--methods", nargs="+",
                   default=["swudi", "aswudi", "wudi2"])
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)

    print(f"Target={args.target} GPU={args.gpu}", flush=True)

    if args.target == "qwen2vl":
        load_fn = load_qwen
        wudi2_fn = WUDI2_QWEN.wudi_merging2
    else:
        load_fn = load_internvl
        wudi2_fn = WUDI2_INTERNVL.wudi_merging2

    print("Loading base + 5 experts ...", flush=True)
    t0 = time.perf_counter()
    base, experts, exclude = load_fn()
    print(f"Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    results = {"target": args.target, "loading_sec": time.perf_counter() - t0,
               "methods": {}}

    if "swudi" in args.methods:
        results["methods"]["swudi_t300_r085"] = time_method(
            "swudi_t300_r085",
            lambda: SWUDI_MOD.swudi_merge(
                base_model=base, finetuned_models=experts,
                exclude_param_names_regex=exclude,
                scaling_coefficient=1.0, merge_device="cuda",
                exp_time=300.0, truncate_rank_ratio=0.85,
                filter_type="exponential", init_mode="sum",
                progress=False,
            ),
        )

    if "aswudi" in args.methods:
        results["methods"]["aswudi_psqrt"] = time_method(
            "aswudi_psqrt",
            lambda: SWUDI_MOD.aswudi_merge(
                base_model=base, finetuned_models=experts,
                exclude_param_names_regex=exclude,
                scaling_coefficient=1.0, merge_device="cuda",
                auto_rank_method="participation_sqrt",
                filter_type="none", init_mode="sum",
                progress=False,
            ),
        )

    if "wudi2" in args.methods:
        results["methods"]["wudi2_optmerge"] = time_method(
            "wudi2_optmerge",
            lambda: wudi2_fn(
                merged_model=base, models_to_merge=experts,
                exclude_param_names_regex=exclude,
                scaling_coefficient=1.0,
            ),
        )

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
