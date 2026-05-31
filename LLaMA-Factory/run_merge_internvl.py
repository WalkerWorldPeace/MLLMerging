"""CLI runner for SWUDI / ASWUDI on InternVL2_5-1B multi-expert merging.

Examples:
    python run_merge_internvl.py --method swudi  --output_path merged/internvl_swudi_t300_r085_s03 \
        --scaling_coefficient 0.3 --truncate_rank_ratio 0.85 --exp_time 300
    python run_merge_internvl.py --method aswudi --output_path merged/internvl_aswudi_psqrt_s03 \
        --scaling_coefficient 0.3 --auto_rank_method participation_sqrt

Default expert pool follows the InternVL test.py / model_merging.py setup:
    base       OpenGVLab/InternVL2_5-1B
    experts    yongxianwei/InternVL2_5-1B_{OCR,VQA,Geometry,Chart,Grounding}

Exclude regex matches the existing
``MLLMerging/InternVL/internvl_chat/model_merging.py`` exactly:
    vision_model.*, .*lm_head.*, .*norm.*, .*embed_tokens.*, .*bias.*

Only LLM body 2-D matrices are merged via SWUDI / ASWUDI; vision tower,
embeddings, lm_head, norms and biases are kept from base.

scaling_coefficient: per Qwen2-VL learnings (mllmerging.md §6.1 / §6.2),
the default 1.0 collapses MLLM outputs to gibberish. Sweet spot is 0.3.
"""

import argparse
import os

import torch
from transformers import AutoModel, AutoTokenizer

from swudi_aswudi import aswudi_merge, swudi_merge


DEFAULT_BASE = "OpenGVLab/InternVL2_5-1B"
DEFAULT_EXPERTS = [
    "yongxianwei/InternVL2_5-1B_OCR",
    "yongxianwei/InternVL2_5-1B_VQA",
    "yongxianwei/InternVL2_5-1B_Geometry",
    "yongxianwei/InternVL2_5-1B_Chart",
    "yongxianwei/InternVL2_5-1B_Grounding",
]
# Mirrors MLLMerging/InternVL/internvl_chat/model_merging.py exclude regex
# (vision tower, embeddings, lm_head, norm, bias kept from base).
DEFAULT_EXCLUDE_REGEX = [
    "vision_model.*",
    ".*lm_head.*",
    ".*norm.*",
    ".*embed_tokens.*",
    ".*bias.*",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["swudi", "aswudi"])
    p.add_argument("--output_path", required=True)
    p.add_argument("--scaling_coefficient", type=float, default=0.3,
                   help="Default 0.3; MLLM full-param delta merging collapses at 1.0.")

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
    p.add_argument("--truncate_rank_ratio", type=float, default=0.85,
                   help="0.65 = CLIP default; 0.85 = Llama/VLM-tuned default.")
    p.add_argument("--filter_type", default=None,
                   choices=[None, "exponential", "none"])
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=True, use_fast=False
    )
    base_model = AutoModel.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True,
    ).eval()

    finetuned = []
    for path in args.experts:
        print(f"Loading expert from {path} ...")
        m = AutoModel.from_pretrained(
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
    tokenizer.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
