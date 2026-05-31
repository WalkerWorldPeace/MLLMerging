"""Precompute and cache the per-layer task-vector stack to avoid reloading
all 9 CLIP-ViT-B/32 checkpoints inside every diagnostic script.

Run once:
    cd "$REPO_ROOT"   # repository root
    python examples/yongxianwei_merging/experiments/cache_task_vectors.py

Output:
    outputs/yongxianwei_merging/theory_diagnostics/task_vectors.pt
    {
        "layer_names": [str, ...],
        "shapes": [(N, m, d), ...],
        "stacks": List[Tensor],
        "pretrained_shapes": [(m, d), ...],
        "pretrained_norms": [float, ...],
    }

The full file is ~200 MB on disk. Saved as float16 to halve that.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (DEFAULT_OUT, get_2d_layer_names, load_pool_state_dicts,
                     load_clip_vision, PRETRAINED_REPO, TASK_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "task_vectors.pt"))
    ap.add_argument("--max_layers", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[cache] {out_path} already exists, skipping")
        return

    pre = load_clip_vision(PRETRAINED_REPO)
    layer_names = get_2d_layer_names(pre, max_layers=args.max_layers)
    pre_state = {n: p.detach().clone() for n, p in pre.state_dict().items()}
    pre_norms = {n: float(pre_state[n].to(torch.float32).norm().item())
                 for n in layer_names if n in pre_state}
    pre_shapes = {n: tuple(pre_state[n].shape) for n in layer_names if n in pre_state}
    del pre

    print(f"[cache] {len(layer_names)} 2-D layers; loading 8 experts", flush=True)
    expert_sds = []
    from _common import EXPERT_REPOS
    for i, repo in enumerate(EXPERT_REPOS):
        t0 = time.time()
        m = load_clip_vision(repo)
        sd = {n: p.detach().clone() for n, p in m.state_dict().items()}
        expert_sds.append(sd)
        del m
        print(f"  [{i+1}/{len(EXPERT_REPOS)}] {repo}  ({time.time()-t0:.1f}s)", flush=True)

    print("[cache] building per-layer task-vector stacks", flush=True)
    stacks = {}
    for name in layer_names:
        if name not in pre_state:
            continue
        base = pre_state[name].to(torch.float32)
        deltas = []
        skip = False
        for sd in expert_sds:
            if name not in sd or sd[name].shape != base.shape:
                skip = True
                break
            deltas.append((sd[name].to(torch.float32) - base).to(torch.float16))
        if skip:
            continue
        stacks[name] = torch.stack(deltas, dim=0)  # (N, m, d) fp16

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "layer_names": list(stacks.keys()),
        "stacks": stacks,
        "pretrained_norms": pre_norms,
        "pretrained_shapes": pre_shapes,
        "task_names": TASK_NAMES,
        "pretrained_repo": PRETRAINED_REPO,
    }, out_path)
    print(f"[cache] wrote {out_path} (n_layers={len(stacks)})")


if __name__ == "__main__":
    main()
