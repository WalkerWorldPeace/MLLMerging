"""Wall-clock timing benchmark for yongxianwei_merging methods.

For each method, runs N trials on a fixed modelpool and reports
mean ± std of the merge-only wall-clock (excludes pretrained / finetuned
loading). Loading is amortized once per modelpool.

Usage:
    python examples/yongxianwei_merging/benchmark_merge_walltime.py \\
        --modelpool clip-vit-base-patch32_TA8_model_only \\
        --methods wudi wudi2 iwudi swudi aswudi \\
        --trials 3 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fusion_bench.method.yongxianwei_merging.functional import (
    dispatch_yongxianwei_merge,
)
from fusion_bench.method.yongxianwei_merging.task_vector import (
    get_param_names_to_merge,
)
from fusion_bench.utils import instantiate

logging.basicConfig(level=logging.WARNING, format="%(message)s")


METHOD_KWARGS = {
    "wudi": {"iter_num": 300, "learning_rate": 1e-5, "progress": False},
    "wudi2": {"iter_num": 300, "learning_rate": 1e-5, "progress": False},
    "iwudi": {"filter_type": "exponential", "exp_time": 300.0, "init_mode": "sum",
              "progress": False},
    "swudi": {"exp_time": 1300.0, "truncate_rank_ratio": 0.65,
              "filter_type": "exponential", "init_mode": "sum", "progress": False},
    "aswudi": {"auto_rank_method": "participation_sqrt", "filter_type": "none",
               "init_mode": "sum", "progress": False},
}


def load_pool_param_dicts(modelpool_name: str, device: str,
                          exclude_param_names_regex=None):
    """Load pretrained + finetuned param dicts from a modelpool config."""
    # Some configs use Hydra `defaults:` composition (e.g. B/16). To avoid the
    # full Hydra runtime, build the config inline for known pools.
    INLINE_POOLS = {
        "clip-vit-base-patch32_TA8_model_only": {
            "ckpt": "openai/clip-vit-base-patch32",
            "task_prefix": "tanganke/clip-vit-base-patch32_",
        },
        "clip-vit-base-patch16_TA8_model_only": {
            "ckpt": "openai/clip-vit-base-patch16",
            "task_prefix": "tanganke/clip-vit-base-patch16_",
        },
    }
    TA8 = ["sun397", "stanford-cars", "resisc45", "eurosat",
           "svhn", "gtsrb", "mnist", "dtd"]

    if modelpool_name in INLINE_POOLS:
        spec = INLINE_POOLS[modelpool_name]
        cfg_dict = {
            "_target_": "fusion_bench.modelpool.CLIPVisionModelPool",
            "_recursive_": False,
            "processor": spec["ckpt"],
            "models": {"_pretrained_": spec["ckpt"], **{
                t: spec["task_prefix"] + t for t in TA8
            }},
        }
        cfg = OmegaConf.create(cfg_dict)
    else:
        cfg_path = Path("config/modelpool/CLIPVisionModelPool") / f"{modelpool_name}.yaml"
        if not cfg_path.exists():
            cfg_path = Path("config/modelpool") / f"{modelpool_name}.yaml"
        cfg = OmegaConf.load(cfg_path)
    modelpool = instantiate(cfg)

    print(f"[load] pretrained")
    pretrained = modelpool.load_model("_pretrained_").to(device)
    pre = {n: p.detach() for n, p in pretrained.named_parameters()}
    del pretrained

    candidate = get_param_names_to_merge(
        list(pre.keys()), exclude_param_names_regex or []
    )
    fine_dicts = []
    for name in modelpool.model_names:
        print(f"[load] {name}")
        m = modelpool.load_model(name).to(device)
        fine_dicts.append({n: p.detach() for n, p in m.named_parameters()})
        del m
        torch.cuda.empty_cache()

    mergeable = [n for n in candidate
                 if pre[n].is_floating_point() and all(
                     n in f and f[n].shape == pre[n].shape for f in fine_dicts
                 )]
    skipped = [n for n in candidate if n not in set(mergeable)]
    if skipped:
        print(f"[load] {len(skipped)} candidate param(s) skipped due to shape "
              f"mismatch or non-float dtype (e.g. {skipped[:3]})")
    pre_subset = {n: pre[n] for n in mergeable}
    fine_subsets = [{n: f[n] for n in mergeable} for f in fine_dicts]
    # Free the discarded unmerged tensors before timing.
    del pre, fine_dicts
    torch.cuda.empty_cache()
    return pre_subset, fine_subsets, len(mergeable)


def time_one(method: str, pre, fine, device, kwargs, exclude_regex=None):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)
    t0 = time.perf_counter()
    out = dispatch_yongxianwei_merge(
        method_name=method,
        base_model=pre,
        finetuned_models=fine,
        exclude_param_names_regex=exclude_regex,
        scaling_coefficient=1.0,
        method_kwargs=dict(kwargs),
        merge_device=torch.device(device),
        progress=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del out
    torch.cuda.empty_cache()
    return {
        "elapsed_s": elapsed,
        "base_alloc_bytes": int(base_alloc),
        "peak_alloc_bytes": int(peak_alloc),
        "peak_reserved_bytes": int(peak_reserved),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelpool", required=True,
                    help="modelpool name relative to config/modelpool/CLIPVisionModelPool/ "
                         "(or path under config/modelpool/, e.g. "
                         "CausalLMPool/mergebench/Llama-3.2-3B)")
    ap.add_argument("--methods", nargs="+",
                    default=["wudi", "wudi2", "iwudi", "swudi", "aswudi"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="JSON output path; auto if not given")
    ap.add_argument("--exclude-regex", nargs="*", default=None,
                    help="Regex patterns of param names to exclude from "
                         "merging (e.g. embed_tokens lm_head).")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip warm-up trial (useful for very expensive merges "
                         "like Llama wudi where one extra run wastes ~2 h).")
    args = ap.parse_args()

    pre, fine, n_layers = load_pool_param_dicts(
        args.modelpool, args.device,
        exclude_param_names_regex=args.exclude_regex,
    )
    print(f"[loaded] {n_layers} mergeable params; "
          f"{len(fine)} finetuned models")

    results = {"modelpool": args.modelpool, "n_mergeable_params": n_layers,
               "n_finetuned": len(fine), "device": args.device,
               "trials": args.trials,
               "exclude_regex": args.exclude_regex,
               "methods": {}}

    for method in args.methods:
        kwargs = METHOD_KWARGS.get(method, {"progress": False})
        # Warm-up trial (excluded from stats; CUDA init / cublas tuning).
        if not args.no_warmup:
            print(f"[warmup] {method}")
            _ = time_one(method, pre, fine, args.device, kwargs,
                         exclude_regex=args.exclude_regex)
        trial_records = []
        for trial in range(args.trials):
            rec = time_one(method, pre, fine, args.device, kwargs,
                           exclude_regex=args.exclude_regex)
            print(f"[{method}] trial {trial+1}/{args.trials}: "
                  f"{rec['elapsed_s']:.2f} s, "
                  f"peak_alloc={rec['peak_alloc_bytes']/1e9:.2f} GB, "
                  f"peak_reserved={rec['peak_reserved_bytes']/1e9:.2f} GB")
            trial_records.append(rec)
        times = [r["elapsed_s"] for r in trial_records]
        peaks = [r["peak_alloc_bytes"] for r in trial_records]
        peaks_res = [r["peak_reserved_bytes"] for r in trial_records]
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        peak_mean = sum(peaks) / len(peaks)
        peak_max = max(peaks)
        peak_res_max = max(peaks_res)
        results["methods"][method] = {
            "trials": trial_records,
            "trials_s": times,
            "mean_s": mean,
            "std_s": std,
            "peak_alloc_bytes_max": int(peak_max),
            "peak_alloc_bytes_mean": int(peak_mean),
            "peak_reserved_bytes_max": int(peak_res_max),
            "kwargs": kwargs,
        }
        print(f"[{method}] mean={mean:.2f} s, std={std:.3f} s, "
              f"peak_alloc_max={peak_max/1e9:.2f} GB, "
              f"peak_reserved_max={peak_res_max/1e9:.2f} GB")

    out = args.out or (
        f"outputs/yongxianwei_merging/timing/"
        f"{args.modelpool.replace('/', '_')}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
