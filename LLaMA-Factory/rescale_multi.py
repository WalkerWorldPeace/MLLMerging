"""Rescale SWUDI/ASWUDI merged checkpoints back toward base at multiple
scales in a single pass. Loads base + merged once per source, then writes
each scaled copy.

Usage:
    python rescale_multi.py \
        --src merged/swudi_t300_r085 \
        --scales 0.3 0.5 0.7 \
        --out_dir merged
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors.torch import load_file, save_file


def load_all_shards(path):
    shards = sorted([f for f in os.listdir(path) if f.endswith(".safetensors")])
    state = {}
    for sh in shards:
        t0 = time.time()
        d = load_file(os.path.join(path, sh))
        state.update(d)
        print(f"  loaded {sh} ({len(d)} tensors, {time.time()-t0:.1f}s)", flush=True)
    return state


def main():
    _hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    _default_base = os.path.join(_hf, "hub", "models--Qwen--Qwen2-VL-7B", "snapshots")
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--base", default=_default_base,
                   help="Qwen2-VL-7B base snapshot dir (default: ${HF_HOME}/hub/models--Qwen--Qwen2-VL-7B/snapshots)")
    p.add_argument("--scales", type=float, nargs="+", required=True)
    p.add_argument("--out_dir", required=True,
                   help="parent dir; outputs go to <out_dir>/<src_basename>_s<NN>")
    args = p.parse_args()

    if os.path.isdir(args.base):
        snaps = [d for d in os.listdir(args.base) if not d.startswith(".")]
        if len(snaps) == 1:
            args.base = os.path.join(args.base, snaps[0])

    src_base = os.path.basename(os.path.normpath(args.src))

    print(f"Loading base from {args.base} ...", flush=True)
    base_state = load_all_shards(args.base)
    print(f"  base: {len(base_state)} tensors", flush=True)

    print(f"Loading merged from {args.src} ...", flush=True)
    src_state = load_all_shards(args.src)
    print(f"  merged: {len(src_state)} tensors", flush=True)

    # Read shard map from src (we keep src's layout, base may have different layout)
    idx_path = os.path.join(args.src, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    keep_files = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "added_tokens.json", "special_tokens_map.json",
        "chat_template.json", "preprocessor_config.json",
        "model.safetensors.index.json",
    ]

    for scale in args.scales:
        tag = f"s{int(round(scale * 10)):02d}"
        dst = os.path.join(args.out_dir, f"{src_base}_{tag}")
        os.makedirs(dst, exist_ok=True)
        print(f"\n=== Building {dst} (scale={scale}) ===", flush=True)

        new_state = {}
        n_blend = 0
        n_copy = 0
        t0 = time.time()
        for k, v_src in src_state.items():
            v_base = base_state.get(k)
            if v_base is None or v_base.shape != v_src.shape:
                new_state[k] = v_src
                n_copy += 1
                continue
            # Linear blend: new = scale*src + (1-scale)*base.
            # Equivalent to base + scale*(src-base).
            # If src and base are bit-equal (excluded params), blend = base = src — same result.
            blended = (scale * v_src.float() + (1.0 - scale) * v_base.float()).to(v_src.dtype)
            new_state[k] = blended
            n_blend += 1
        print(f"  blended {n_blend}, copied {n_copy} in {time.time()-t0:.1f}s", flush=True)

        # Save in src's shard layout
        shard_to_tensors = {}
        for name, sh in weight_map.items():
            shard_to_tensors.setdefault(sh, {})[name] = new_state[name]

        t0 = time.time()
        for sh, tensors in shard_to_tensors.items():
            save_file(tensors, os.path.join(dst, sh), metadata={"format": "pt"})
        print(f"  wrote {len(shard_to_tensors)} shards in {time.time()-t0:.1f}s", flush=True)

        # Copy non-weight files
        for f in keep_files:
            src_f = os.path.join(args.src, f)
            if os.path.exists(src_f):
                shutil.copy(src_f, os.path.join(dst, f))

        # Free new_state for next iter
        del new_state
        del shard_to_tensors

    print("\nAll scales done.", flush=True)


if __name__ == "__main__":
    main()
