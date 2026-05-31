"""CLI runner for SWUDI / ASWUDI on Qwen2-VL multi-expert merging.

Examples:
    python run_merge.py --method swudi  --output_path merged/swudi_default
    python run_merge.py --method swudi  --truncate_rank_ratio 0.85 --exp_time 300 \
                        --output_path merged/swudi_t300_r085
    python run_merge.py --method aswudi --auto_rank_method participation_sqrt \
                        --output_path merged/aswudi_psqrt
    python run_merge.py --method aswudi --auto_rank_method gavish_donoho \
                        --output_path merged/aswudi_gd

Default expert pool follows the OptMerge paper:
    base       Qwen/Qwen2-VL-7B
    experts    yongxianwei/Qwen2-VL-7B-{OCR,VQA,Geometry,Chart,Grounding}

The exclude_param_names_regex is identical to the existing
``MLLMerging/LLaMA-Factory/model_merging.py`` (visual.*, embed_tokens,
lm_head, norm, bias) -- only the LLM body is merged.
"""

import argparse
import os

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from swudi_aswudi import aswudi_merge, swudi_merge


DEFAULT_BASE = "Qwen/Qwen2-VL-7B"
DEFAULT_EXPERTS = [
    "yongxianwei/Qwen2-VL-7B-OCR",
    "yongxianwei/Qwen2-VL-7B-VQA",
    "yongxianwei/Qwen2-VL-7B-Geometry",
    "yongxianwei/Qwen2-VL-7B-Chart",
    "yongxianwei/Qwen2-VL-7B-Grounding",
]
DEFAULT_EXCLUDE_REGEX = [
    "visual..*",
    ".*embed_tokens.*",
    ".*lm_head.*",
    ".*norm.*",
    ".*bias.*",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["swudi", "aswudi"])
    p.add_argument("--output_path", required=True)
    p.add_argument("--scaling_coefficient", type=float, default=1.0)

    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument(
        "--experts", nargs="+", default=DEFAULT_EXPERTS,
        help="HuggingFace repo ids or local paths for fine-tuned experts.",
    )
    p.add_argument(
        "--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"],
    )

    # SWUDI knobs
    p.add_argument("--exp_time", type=float, default=300.0)
    p.add_argument("--truncate_rank_ratio", type=float, default=0.65,
                   help="SWUDI hard-truncation ratio. 0.65 = CLIP default; "
                        "0.85 = Llama-tuned default. Set <=0 to disable.")
    p.add_argument("--filter_type", default=None,
                   choices=[None, "exponential", "none"],
                   help="None -> use the method's default "
                        "(swudi: exponential, aswudi: none).")
    p.add_argument("--init_mode", default="sum", choices=["sum", "mean", "zero"])

    # ASWUDI knobs
    p.add_argument("--auto_rank_method", default="participation_sqrt",
                   help="participation_sqrt | gavish_donoho | entropy | "
                        "stable_rank | participation | cumvar_<frac> | none")
    p.add_argument("--rank_scale", type=float, default=1.0)
    p.add_argument("--max_rank_ratio", type=float, default=1.0)

    p.add_argument("--merge_device", default="cuda")
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.torch_dtype]

    print(f"Loading base from {args.base} ...")
    processor = AutoProcessor.from_pretrained(args.base)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True,
    ).eval()

    finetuned = []
    for path in args.experts:
        print(f"Loading expert from {path} ...")
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True,
        ).eval()
        finetuned.append(m)

    common = dict(
        exclude_param_names_regex=DEFAULT_EXCLUDE_REGEX,
        scaling_coefficient=float(args.scaling_coefficient),
        merge_device=args.merge_device,
        init_mode=args.init_mode,
        progress=not args.no_progress,
    )

    if args.method == "swudi":
        ratio = args.truncate_rank_ratio if args.truncate_rank_ratio > 0 else None
        merged = swudi_merge(
            base_model=base_model,
            finetuned_models=finetuned,
            exp_time=float(args.exp_time),
            truncate_rank_ratio=ratio,
            filter_type=args.filter_type or "exponential",
            **common,
        )
    else:  # aswudi
        merged = aswudi_merge(
            base_model=base_model,
            finetuned_models=finetuned,
            auto_rank_method=args.auto_rank_method,
            rank_scale=float(args.rank_scale),
            max_rank_ratio=float(args.max_rank_ratio),
            filter_type=args.filter_type or "none",
            exp_time=float(args.exp_time),
            **common,
        )

    base_state = base_model.state_dict()
    for k, v in merged.items():
        if k in base_state:
            base_state[k] = v
    base_model.load_state_dict(base_state)

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving merged model to {args.output_path} ...")
    base_model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
