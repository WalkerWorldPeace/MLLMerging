"""Linearly blend a SWUDI/ASWUDI merged checkpoint back toward the base
Qwen2-VL-7B at a chosen scaling_coefficient, without re-running the merge.

For every parameter:
    new = scale * merged + (1 - scale) * base
which is identically:
    new = base + scale * (merged - base)
i.e. base plus a re-scaled version of the merged delta.

Excluded params (visual.*, embed_tokens, lm_head, norm.*, bias) are
unchanged between base and merged, so the blend is a no-op for them.

Usage:
    python rescale_merged.py --src merged/swudi_t300_r085 --scale 0.3 \
        --dst merged/swudi_t300_r085_s030
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor


def main():
    _hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    _default_base = os.path.join(_hf, "hub", "models--Qwen--Qwen2-VL-7B", "snapshots")
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="merged checkpoint path")
    p.add_argument("--base", default=_default_base,
                   help="Qwen2-VL-7B base snapshot dir (default: ${HF_HOME}/hub/models--Qwen--Qwen2-VL-7B/snapshots)")
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--dst", required=True, help="output checkpoint path")
    args = p.parse_args()

    if os.path.isdir(args.base):
        # Pick the actual snapshot dir
        snaps = [d for d in os.listdir(args.base) if not d.startswith(".")]
        if len(snaps) == 1:
            args.base = os.path.join(args.base, snaps[0])

    os.makedirs(args.dst, exist_ok=True)

    # Find safetensor shards in src and base
    src_shards = sorted([f for f in os.listdir(args.src) if f.endswith(".safetensors")])
    base_shards = sorted([f for f in os.listdir(args.base) if f.endswith(".safetensors")])
    assert src_shards and base_shards, "no shards found"
    print(f"src shards: {src_shards}")
    print(f"base shards: {base_shards}")
    print(f"scale: {args.scale}")

    # Build an index name -> (src_shard_path, tensor_dict_lazy)
    # We'll iterate base shards and for each tensor, look up the same name in src.
    # Most efficient: load base shards once, load src shards once.

    print("Loading base shards ...")
    base_state = {}
    for sh in base_shards:
        d = load_file(os.path.join(args.base, sh))
        base_state.update(d)
    print(f"  base has {len(base_state)} tensors")

    print("Loading src (merged) shards ...")
    src_state = {}
    for sh in src_shards:
        d = load_file(os.path.join(args.src, sh))
        src_state.update(d)
    print(f"  src has {len(src_state)} tensors")

    print(f"Blending: new = scale * merged + (1-scale) * base, scale={args.scale}")
    new_state = {}
    n_changed = 0
    n_unchanged = 0
    for k in src_state:
        v_src = src_state[k]
        if k in base_state:
            v_base = base_state[k]
            if v_src.shape != v_base.shape:
                print(f"  shape mismatch {k}: src={v_src.shape}, base={v_base.shape}; keep src")
                new_state[k] = v_src
                continue
            if torch.equal(v_src, v_base):
                new_state[k] = v_base
                n_unchanged += 1
            else:
                blended = (args.scale * v_src.float() + (1.0 - args.scale) * v_base.float()).to(v_src.dtype)
                new_state[k] = blended
                n_changed += 1
        else:
            print(f"  src-only key (kept as-is): {k}")
            new_state[k] = v_src
    print(f"  changed (delta scaled): {n_changed}, unchanged (excluded or identical): {n_unchanged}")

    # Save in same shard layout as src
    print("Saving new shards ...")
    # Read shard map from src's index
    import json
    idx_path = os.path.join(args.src, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]   # name -> shard file
    # Group by shard
    shard_to_tensors = {}
    for name, sh in weight_map.items():
        shard_to_tensors.setdefault(sh, {})[name] = new_state[name]

    for sh, tensors in shard_to_tensors.items():
        save_file(tensors, os.path.join(args.dst, sh), metadata={"format": "pt"})
        print(f"  wrote {sh} ({len(tensors)} tensors)")

    # Copy non-weight files (config.json, generation_config.json, tokenizer*, processor*, chat_template, index)
    keep_files = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "added_tokens.json", "special_tokens_map.json",
        "chat_template.json", "preprocessor_config.json",
        "model.safetensors.index.json",
    ]
    for f in keep_files:
        src_f = os.path.join(args.src, f)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(args.dst, f))
    print(f"Done. Output at {args.dst}")


if __name__ == "__main__":
    main()
