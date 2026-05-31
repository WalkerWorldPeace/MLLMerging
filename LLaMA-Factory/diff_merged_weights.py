"""Diff merged Qwen2-VL checkpoints: NaN/inf scan, L2-norm comparison.

Compares optmerge_s10 vs aswudi_psqrt_s02 vs Qwen base — to find
whether OptMerge merge produced bad weights.
"""
import os
import sys
import torch
from safetensors.torch import load_file


def load_all(path):
    state = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            state.update(load_file(os.path.join(path, f)))
    return state


paths = {
    # Default Qwen2-VL-7B base snapshot location: ${HF_HOME}/hub/...; override
    # via argparse --base or by setting QWEN_BASE_SNAP / ASWUDI_PATH / OPTMERGE_PATH.
    "base":     os.environ.get(
        "QWEN_BASE_SNAP",
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub", "models--Qwen--Qwen2-VL-7B", "snapshots",
        ),
    ),
    "aswudi":   os.environ["ASWUDI_PATH"]   if "ASWUDI_PATH"   in os.environ else None,
    "optmerge": os.environ["OPTMERGE_PATH"] if "OPTMERGE_PATH" in os.environ else None,
}

# Resolve base snapshot
snaps = [d for d in os.listdir(paths["base"]) if not d.startswith(".")]
paths["base"] = os.path.join(paths["base"], snaps[0])

states = {}
for k, p in paths.items():
    print(f"Loading {k} from {p}", flush=True)
    states[k] = load_all(p)
    print(f"  {len(states[k])} tensors", flush=True)

# Check key sets identical
ks_base = set(states["base"].keys())
ks_aswudi = set(states["aswudi"].keys())
ks_opt = set(states["optmerge"].keys())
print(f"\nKey set diffs:")
print(f"  base ^ aswudi: {len(ks_base ^ ks_aswudi)}")
print(f"  base ^ optmerge: {len(ks_base ^ ks_opt)}")
print(f"  aswudi ^ optmerge: {len(ks_aswudi ^ ks_opt)}")

# Check NaN / Inf
for tag in ["aswudi", "optmerge"]:
    nan_keys, inf_keys = [], []
    for k, v in states[tag].items():
        if not v.is_floating_point():
            continue
        if torch.isnan(v).any():
            nan_keys.append(k)
        if torch.isinf(v).any():
            inf_keys.append(k)
    print(f"\n{tag}: NaN keys={len(nan_keys)}, Inf keys={len(inf_keys)}")
    if nan_keys[:3]:
        print(f"  first NaN keys: {nan_keys[:3]}")
    if inf_keys[:3]:
        print(f"  first Inf keys: {inf_keys[:3]}")

# Compare L2 of (merged - base) for a few representative layers
def lookup(state, name):
    return state.get(name)

print("\nLayer delta vs base (||v - base||_2):")
sample_keys = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.27.mlp.gate_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
]
for k in sample_keys:
    b = lookup(states["base"], k)
    a = lookup(states["aswudi"], k)
    o = lookup(states["optmerge"], k)
    if b is None:
        print(f"  [{k}] not in base, skip")
        continue
    if a is not None and o is not None:
        d_a = (a.float() - b.float()).norm().item()
        d_o = (o.float() - b.float()).norm().item()
        print(f"  [{k}] aswudi-Δ={d_a:.4f}  optmerge-Δ={d_o:.4f}  base‖={b.float().norm().item():.4f}")
    else:
        print(f"  [{k}] missing in some merged set")

# Quick global stat: total Frobenius norm of (merged - base) for matched 2D params
def total_delta(merged_state, base_state):
    total = 0.0
    n = 0
    max_layer = ""
    max_d = 0.0
    for k, v in merged_state.items():
        if k in base_state and v.shape == base_state[k].shape and v.is_floating_point() and v.dim() == 2 and "lm_head" not in k:
            d = (v.float() - base_state[k].float()).norm().item()
            total += d
            n += 1
            if d > max_d:
                max_d = d
                max_layer = k
    return total, n, max_d, max_layer

print()
for tag in ["aswudi", "optmerge"]:
    total, n, mx, ml = total_delta(states[tag], states["base"])
    print(f"{tag}: sum_2D ||Δ|| = {total:.2f} over {n} layers; max layer={ml} ({mx:.4f})")
