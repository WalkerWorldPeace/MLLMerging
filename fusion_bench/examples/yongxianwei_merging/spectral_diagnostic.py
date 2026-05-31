"""Per-layer spectral diagnostic for task-vector matrices.

For each modelpool (CLIP-ViT-B/32 TA8, Flan-T5 GLUE LoRA r=16, Llama-3.2-3B
MergeBench), compute quantities that are mechanistically predicted by the
Δ-structure analysis in iwudi_paper.md §5.10.

For every 2-D layer shared across the N experts we build

    v_i = W_finetuned_i - W_base         # (m x d)
    C   = Σ_i (v_i^T v_i) / ||v_i||_F^2  # (d x d)

and compute eigvals(C) via torch.linalg.eigh (same numerical path as SWUDI's
functional.py). Per layer we report:

  r_eff            = (Σ λ)^2 / Σ λ^2               # IPR effective rank
  r_eff_norm       = r_eff / d                     # normalized effective rank
  peak_ratio       = λ_max / λ_median
  decay_tail_ratio = λ_{0.9*d} / λ_max             # 90th percentile over max
  cumvar_at_r      = Σ_{i<=r*d} λ_i / Σ λ_i  for r in {0.50, 0.65, 0.85}
  lambda_max       = λ_max                         # for layer-to-layer heterogeneity
  frob_ratio       = mean_i ||v_i||_F / ||W_base||_F   # relative magnitude

We run each pool on one GPU. Outputs land in
    outputs/yongxianwei_merging/spectral/<pool>/per_layer.json
    outputs/yongxianwei_merging/spectral/<pool>/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _load_state_dict(repo_id: str) -> Dict[str, torch.Tensor]:
    """Load a state_dict from an HF repo without going through transformers.from_pretrained
    (which blocks .bin files under CVE-2025-32434 when torch < 2.6)."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id,
        allow_patterns=[
            "*.safetensors", "*.safetensors.index.json",
            "*.bin", "pytorch_model.bin.index.json",
        ],
    )
    local = Path(local_dir)
    state: Dict[str, torch.Tensor] = {}

    # Prefer safetensors, fall back to .bin
    safetensors_files = sorted(local.glob("*.safetensors"))
    if safetensors_files:
        from safetensors.torch import load_file
        for sf in safetensors_files:
            state.update(load_file(str(sf), device="cpu"))
        return state

    # Bin fallback — use torch.load directly with weights_only=True (safe as long as
    # trusted HF repos; we still pin torch==2.5.1 and transformers' extra check is
    # just a version gate on the loader API, not the actual safety mechanism).
    bin_files = sorted(local.glob("*.bin"))
    for bf in bin_files:
        state.update(torch.load(str(bf), map_location="cpu", weights_only=True))
    return state


def _load_peft_adapter_deltas(peft_id: str) -> Dict[str, torch.Tensor]:
    """Load a LoRA adapter without going through peft.PeftModel (which requires a
    compatible base model + modern huggingface_hub). Returns a dict mapping base
    parameter name -> Δ = (lora_B @ lora_A) * scaling.

    Key naming in peft saved adapters:
        base_model.model.<...>.<layer>.lora_A.weight  (shape: r, in)
        base_model.model.<...>.<layer>.lora_B.weight  (shape: out, r)

    We strip the ``base_model.model.`` prefix and append ``.weight`` to match the
    original base model's parameter names (e.g. decoder.block.0.layer.0.SelfAttention.q.weight).
    """
    import json
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(peft_id, "adapter_config.json")
    cfg = json.load(open(cfg_path))
    r = float(cfg.get("r", 16))
    alpha = float(cfg.get("lora_alpha", r))
    use_rslora = cfg.get("use_rslora", False)
    if use_rslora:
        import math as _math
        scaling = alpha / _math.sqrt(r)
    else:
        scaling = alpha / r

    state = _load_state_dict(peft_id)
    # Group into (base_name, {"A": ..., "B": ...})
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in state.items():
        if not k.startswith("base_model.model."):
            continue
        body = k[len("base_model.model."):]
        if ".lora_A.weight" in body:
            base_name = body.replace(".lora_A.weight", ".weight")
            pairs.setdefault(base_name, {})["A"] = v.to(torch.float32)
        elif ".lora_B.weight" in body:
            base_name = body.replace(".lora_B.weight", ".weight")
            pairs.setdefault(base_name, {})["B"] = v.to(torch.float32)
    deltas: Dict[str, torch.Tensor] = {}
    for name, AB in pairs.items():
        if "A" not in AB or "B" not in AB:
            continue
        # B: (out, r), A: (r, in)  =>  Δ = B @ A  has shape (out, in)
        deltas[name] = AB["B"] @ AB["A"] * scaling
    return deltas


def _dotenv_load(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def _is_2d_optimizable(param: torch.Tensor, name: str) -> bool:
    return param.ndim == 2 and "lm_head" not in name


def _percentile(vals: torch.Tensor, q: float) -> float:
    """q in [0, 1]; vals 1-D sorted ascending."""
    n = vals.numel()
    if n == 0:
        return float("nan")
    idx = max(0, min(n - 1, int(q * (n - 1))))
    return float(vals[idx].item())


def _layer_spectrum(v_stack: torch.Tensor) -> Optional[Dict[str, float]]:
    """v_stack: (N, m, d) float32/float64. Returns per-layer scalar stats, or None if
    every expert has zero delta on this layer (e.g. LoRA that doesn't touch it)."""
    N, m, d = v_stack.shape
    # norm-squared per expert, clamped to avoid /0
    fro_sq = (v_stack.reshape(N, -1) ** 2).sum(dim=1)  # (N,)
    # Drop experts with zero delta on this layer (they carry no information)
    active = fro_sq > 1e-20
    if active.sum().item() < 2:
        return None
    v_stack = v_stack[active]
    fro_sq = fro_sq[active]
    N = v_stack.shape[0]
    inv_norms = 1.0 / fro_sq.clamp(min=1e-30)                           # (N,)

    # C = Σ_i (v_i^T v_i) / ||v_i||_F^2  — exactly as in functional.py
    # pick the smaller side so eigh runs on min(m, d) x min(m, d)
    if d <= m:
        B = torch.einsum("nab,nac->nbc", v_stack, v_stack)
        C = torch.einsum("n,nbc->bc", inv_norms, B)
    else:
        B = torch.einsum("nab,ncb->nac", v_stack, v_stack)
        C = torch.einsum("n,nac->ac", inv_norms, B)
    eigvals = torch.linalg.eigvalsh(C)            # ascending
    eigvals = eigvals.clamp(min=0.0)
    eigvals_desc = torch.flip(eigvals, dims=[0])  # descending

    total = eigvals.sum().item()
    ssum = (eigvals * eigvals).sum().item()
    if total <= 0 or ssum <= 0:
        return None
    r_eff = (total * total) / ssum
    lam_max = float(eigvals_desc[0].item())
    if lam_max <= 0:
        return None

    # peak_ratio via 50th percentile of ACTIVE eigenvalues (> 0), not all eigvals
    active_eig = eigvals_desc[eigvals_desc > 1e-20 * lam_max]
    if active_eig.numel() == 0:
        return None
    lam_median_active = float(active_eig[active_eig.numel() // 2].item())
    peak_ratio = lam_max / max(lam_median_active, 1e-30)

    # decay tail: 90th percentile within active eigvals / lam_max (closer to 1 = flat)
    n_act = active_eig.numel()
    idx90 = min(n_act - 1, int(0.9 * (n_act - 1)))
    lam90 = float(active_eig[idx90].item())
    decay_tail_ratio = lam90 / max(lam_max, 1e-30)

    # cumulative variance at r * min(m,d)  (r in {0.50, 0.65, 0.85})
    # Use full eigvals_desc (including zeros) so cumvar_at_r reflects actual structure
    n_eig = eigvals_desc.numel()
    cumsum = torch.cumsum(eigvals_desc, dim=0) / max(total, 1e-30)
    cumvar = {}
    for r in (0.50, 0.65, 0.85):
        idx = min(n_eig - 1, max(0, int(round(r * n_eig)) - 1))
        cumvar[f"cumvar_at_{r:.2f}"] = float(cumsum[idx].item())

    mean_fro = float(fro_sq.sqrt().mean().item())
    effective_rank_count = int(n_act)  # how many eigenvalues survive noise floor

    return {
        "m": int(m),
        "d": int(d),
        "rank": int(min(m, d)),
        "n_experts": int(N),
        "lambda_max": lam_max,
        "lambda_median_active": lam_median_active,
        "peak_ratio": peak_ratio,
        "decay_tail_ratio": decay_tail_ratio,
        "r_eff": r_eff,
        "r_eff_norm": r_eff / max(min(m, d), 1),
        "effective_rank_count": effective_rank_count,
        "mean_expert_fro": mean_fro,
        **cumvar,
    }


def _build_layer_deltas(
    base_state: Dict[str, torch.Tensor],
    expert_states: List[Tuple[str, Dict[str, torch.Tensor]]],
    max_params: Optional[int],
    skip_regex: str,
) -> Dict[str, Dict]:
    """Shared logic: given base state + list of (id, expert state), compute per-layer
    (N, m, d) delta stacks for all 2-D non-excluded layers.

    We intersect (a) 2-D non-excluded keys in base with (b) the set of keys present
    in ALL experts with matching shape. Any expert missing a key is dropped from
    that layer; layers shared by fewer than 2 experts are dropped entirely."""
    skip_re = re.compile(skip_regex)
    # Step 1: base-side candidate names
    base_candidates: List[str] = []
    base_frob: Dict[str, float] = {}
    for n, p in base_state.items():
        if _is_2d_optimizable(p, n) and not skip_re.search(n):
            base_candidates.append(n)
            base_frob[n] = float(p.to(torch.float32).norm().item())

    # Step 2: intersect with each expert
    shared = [n for n in base_candidates
              if all(n in st and st[n].shape == base_state[n].shape
                     for _, st in expert_states)]
    print(f"  base 2-D candidates: {len(base_candidates)}  shared with all experts: {len(shared)}",
          flush=True)
    if max_params is not None:
        shared = shared[:max_params]

    layer_deltas: Dict[str, List[torch.Tensor]] = {}
    for _ex_id, ex_state in expert_states:
        for n in shared:
            delta = ex_state[n].to(torch.float32) - base_state[n].to(torch.float32)
            layer_deltas.setdefault(n, []).append(delta)

    return _stack_deltas(layer_deltas, base_frob)


def _load_clip_deltas(
    base_id: str,
    expert_ids: List[str],
    max_params: Optional[int],
    skip_regex: str,
) -> Dict[str, Dict]:
    print(f"  loading base CLIP {base_id}", flush=True)
    base_state = _load_state_dict(base_id)
    # CLIP-ViT repos store the full CLIPModel weights; MergeBench-style vision experts
    # store just CLIPVisionModel. We only keep parameters whose name appears in all
    # experts, so the _build_layer_deltas intersection step handles this automatically.
    expert_states: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    for ex in expert_ids:
        print(f"  loading expert {ex}", flush=True)
        expert_states.append((ex, _load_state_dict(ex)))
    return _build_layer_deltas(base_state, expert_states, max_params, skip_regex)


def _load_causal_lm_deltas(
    base_id: str,
    expert_ids: List[str],
    max_params: Optional[int],
    skip_regex: str,
) -> Dict[str, Dict]:
    print(f"  loading base causal LM {base_id}", flush=True)
    base_state = _load_state_dict(base_id)
    expert_states: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    for ex in expert_ids:
        print(f"  loading expert {ex}", flush=True)
        expert_states.append((ex, _load_state_dict(ex)))
    return _build_layer_deltas(base_state, expert_states, max_params, skip_regex)


def _load_seq2seq_lora_deltas(
    base_id: str,
    lora_ids: List[str],
    max_params: Optional[int],
    skip_regex: str,
) -> Dict[str, Dict]:
    """For LoRA adapters, compute Δ = BA·scaling directly from adapter weights.
    We do NOT load the base model at all for the shape/delta computation; the base
    Frobenius norm is optional and only affects the fro_ratio summary. We load it
    once to get that stat.
    """
    skip_re = re.compile(skip_regex)
    print(f"  loading base seq2seq (only for ||W||_F reference) {base_id}", flush=True)
    base_state = _load_state_dict(base_id)

    adapter_deltas: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    for lora_id in lora_ids:
        print(f"  loading LoRA adapter {lora_id}", flush=True)
        adapter_deltas.append((lora_id, _load_peft_adapter_deltas(lora_id)))

    # Candidate names: only keys that every adapter actually has
    shared: List[str] = []
    first = adapter_deltas[0][1]
    for name in first.keys():
        if not all(name in d for _, d in adapter_deltas):
            continue
        sample_shape = first[name].shape
        if not all(d[name].shape == sample_shape for _, d in adapter_deltas):
            continue
        if skip_re.search(name):
            continue
        shared.append(name)
    print(f"  LoRA layers (appear in ALL {len(lora_ids)} adapters): {len(shared)}",
          flush=True)
    if max_params is not None:
        shared = shared[:max_params]

    base_frob: Dict[str, float] = {}
    for n in shared:
        if n in base_state:
            base_frob[n] = float(base_state[n].to(torch.float32).norm().item())
        else:
            base_frob[n] = float("nan")

    layer_deltas: Dict[str, List[torch.Tensor]] = {}
    for _lora_id, d in adapter_deltas:
        for n in shared:
            layer_deltas.setdefault(n, []).append(d[n])

    return _stack_deltas(layer_deltas, base_frob)


def _stack_deltas(
    layer_deltas: Dict[str, List[torch.Tensor]],
    base_frob: Dict[str, float],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Keep deltas on CPU; return name -> {stack, base_frob}."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for n, deltas in layer_deltas.items():
        if len(deltas) < 2:
            continue
        out[n] = {
            "stack": torch.stack(deltas, dim=0),  # (N, m, d) cpu float32
            "base_frob": base_frob.get(n, float("nan")),
        }
    return out


def _summarise(per_layer: List[Dict]) -> Dict[str, float]:
    def _gather(key: str) -> List[float]:
        return [row[key] for row in per_layer if row.get(key) == row.get(key)]

    import statistics as st

    def _describe(key: str, vs: List[float]) -> Dict[str, float]:
        if not vs:
            return {}
        res = {
            f"{key}_mean": st.fmean(vs),
            f"{key}_median": st.median(vs),
            f"{key}_stdev": st.stdev(vs) if len(vs) > 1 else 0.0,
            f"{key}_min": min(vs),
            f"{key}_max": max(vs),
        }
        if f"{key}_mean" in res and res[f"{key}_mean"] != 0:
            res[f"{key}_cv"] = res[f"{key}_stdev"] / res[f"{key}_mean"]
        return res

    keys = [
        "lambda_max", "peak_ratio", "r_eff", "r_eff_norm",
        "decay_tail_ratio", "cumvar_at_0.50", "cumvar_at_0.65", "cumvar_at_0.85",
        "mean_expert_fro",
    ]
    summary: Dict[str, float] = {"n_layers": len(per_layer)}
    for k in keys:
        summary.update(_describe(k, _gather(k)))

    fro_ratios = [row["mean_expert_fro"] / row["base_frob"]
                  for row in per_layer
                  if row.get("base_frob", 0) > 0 and row.get("mean_expert_fro", 0) > 0]
    summary.update(_describe("fro_ratio", fro_ratios))

    return summary


POOLS = {
    "llama32_3b": {
        "type": "causal_lm",
        "base": "meta-llama/Llama-3.2-3B",
        "experts": [
            "MergeBench/Llama-3.2-3B_instruction",
            "MergeBench/Llama-3.2-3B_math",
            "MergeBench/Llama-3.2-3B_coding",
            "MergeBench/Llama-3.2-3B_multilingual",
            "MergeBench/Llama-3.2-3B_safety",
        ],
        "skip_regex": r"(embed_tokens|lm_head)",
    },
    "clip_vit_b32_ta8": {
        "type": "clip_vision",
        "base": "openai/clip-vit-base-patch32",
        "experts": [
            "tanganke/clip-vit-base-patch32_sun397",
            "tanganke/clip-vit-base-patch32_stanford-cars",
            "tanganke/clip-vit-base-patch32_resisc45",
            "tanganke/clip-vit-base-patch32_eurosat",
            "tanganke/clip-vit-base-patch32_svhn",
            "tanganke/clip-vit-base-patch32_gtsrb",
            "tanganke/clip-vit-base-patch32_mnist",
            "tanganke/clip-vit-base-patch32_dtd",
        ],
        "skip_regex": r"^$",  # CLIP-ViT no exclusion
    },
    "flan_t5_glue_lora16": {
        "type": "seq2seq_lora",
        "base": "google/flan-t5-base",
        "experts": [
            "tanganke/flan-t5-base_glue-cola_lora-16",
            "tanganke/flan-t5-base_glue-mnli_lora-16",
            "tanganke/flan-t5-base_glue-mrpc_lora-16",
            "tanganke/flan-t5-base_glue-qnli_lora-16",
            "tanganke/flan-t5-base_glue-qqp_lora-16",
            "tanganke/flan-t5-base_glue-rte_lora-16",
            "tanganke/flan-t5-base_glue-sst2_lora-16",
            "tanganke/flan-t5-base_glue-stsb_lora-16",
        ],
        "skip_regex": r"^$",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS.keys()))
    ap.add_argument("--out_dir", default="outputs/yongxianwei_merging/spectral")
    ap.add_argument("--max_params", type=int, default=None, help="cap number of layers for debugging")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _dotenv_load(Path(".env.local"))

    spec = POOLS[args.pool]
    out_dir = Path(args.out_dir) / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[{args.pool}] device={device}", flush=True)

    t0 = time.time()
    if spec["type"] == "causal_lm":
        deltas = _load_causal_lm_deltas(spec["base"], spec["experts"],
                                        args.max_params, spec["skip_regex"])
    elif spec["type"] == "clip_vision":
        deltas = _load_clip_deltas(spec["base"], spec["experts"],
                                   args.max_params, spec["skip_regex"])
    elif spec["type"] == "seq2seq_lora":
        deltas = _load_seq2seq_lora_deltas(spec["base"], spec["experts"],
                                           args.max_params, spec["skip_regex"])
    else:
        raise ValueError(spec["type"])
    print(f"[{args.pool}] loaded {len(deltas)} layers in {time.time() - t0:.1f}s", flush=True)

    per_layer: List[Dict] = []
    skipped = 0
    t1 = time.time()
    for i, (name, pack) in enumerate(deltas.items()):
        stack = pack["stack"].to(device)
        stats = _layer_spectrum(stack)
        if stats is None:
            skipped += 1
            del stack
            continue
        stats["name"] = name
        stats["base_frob"] = pack["base_frob"]
        per_layer.append(stats)
        del stack
        if (i + 1) % 20 == 0:
            print(f"  [{args.pool}] {i + 1}/{len(deltas)}  "
                  f"(kept {len(per_layer)}, skipped {skipped}, "
                  f"elapsed {time.time() - t1:.1f}s)", flush=True)
    print(f"[{args.pool}] spectral pass done in {time.time() - t1:.1f}s; "
          f"kept {len(per_layer)} / {len(deltas)} layers (skipped {skipped} zero-delta)",
          flush=True)

    (out_dir / "per_layer.json").write_text(json.dumps(per_layer, indent=2))
    summary = _summarise(per_layer)
    summary["pool"] = args.pool
    summary["n_experts"] = len(spec["experts"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[{args.pool}] wrote {out_dir}/per_layer.json  (n_layers={summary['n_layers']})", flush=True)
    # concise summary
    for k in ("r_eff_norm_mean", "peak_ratio_mean", "peak_ratio_median",
              "cumvar_at_0.65_mean", "cumvar_at_0.85_mean",
              "lambda_max_cv", "fro_ratio_mean"):
        if k in summary:
            print(f"    {k} = {summary[k]:.4g}")


if __name__ == "__main__":
    main()
