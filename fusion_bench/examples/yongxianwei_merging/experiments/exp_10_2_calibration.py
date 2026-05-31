"""§10.2 — Proxy-to-interference calibration.

Two parts:

(A) Proxy-only Pareto: for each merge variant
        {sum, closed_form, IWUDI(t=300), SWUDI(r=0.65), ASWUDI(√λ)}
    we compute the per-layer WUDI proxy P(θ) = Σ_i ||(θ-v_i)v_i^T||²/||v_i||²
    and stack into a single Pareto plot. This is fully data-free.

(B) Real-interference subset: for a small calibration sample of CLIP-ViT-B/32
    images (32 per task × 3 tasks = 96 samples) we collect the input
    activation x_{i,ℓ} at three representative MLP fc1 layers and compute
    the empirical interference
        Î(θ) = Σ_i (1/M_i) Σ_m ||(θ - v_i) x_{i,ℓ,m}||²
    for each merge variant. This measures whether proxy and real interference
    rank methods consistently.

Output:
    outputs/yongxianwei_merging/theory_diagnostics/exp_10_2_proxy_calibration/
        proxy_only.json      (per-layer P(θ) for each method, all 72 layers)
        real_subset.json     (per-layer Î(θ) and P(θ) on subset, with per-method ranking)
        summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (DEFAULT_OUT, HF_CACHE, PRETRAINED_REPO, EXPERT_REPOS,
                     TASK_NAMES,
                     compute_C_D, eigh_descending, exponential_filter_theta,
                     hard_truncation_theta, closed_form_theta,
                     load_clip_vision, write_json,
                     configure_torch_for_diagnostics,
                     participation_sqrt_rank)


# ---------------------------------------------------------------------------
# Part A: Proxy-only across all layers
# ---------------------------------------------------------------------------

def wudi_proxy_layer(theta: torch.Tensor, v_stack: torch.Tensor) -> float:
    eps = 1e-12
    flat = v_stack.reshape(v_stack.shape[0], -1)
    l2 = (flat * flat).sum(dim=-1).clamp(min=eps)
    diff = theta.unsqueeze(0) - v_stack
    prod = torch.matmul(diff, v_stack.transpose(1, 2))
    return float((prod.square().sum(dim=(1, 2)) / l2).sum().item())


def all_method_thetas(v_stack: torch.Tensor):
    """Build θ for each method per layer."""
    C, D, l2_sq, theta0 = compute_C_D(v_stack)
    lam, V = eigh_descending(C)
    d = lam.numel()

    methods = {}
    methods["sum"] = theta0
    methods["closed_form"] = closed_form_theta(theta0, D, lam, V)
    methods["iwudi_t10"] = exponential_filter_theta(theta0, D, lam, V, 10.0)
    methods["iwudi_t100"] = exponential_filter_theta(theta0, D, lam, V, 100.0)
    methods["iwudi_t300"] = exponential_filter_theta(theta0, D, lam, V, 300.0)
    K_swudi = max(1, int(math.ceil(0.65 * d)))
    methods["swudi_r0_65"] = hard_truncation_theta(theta0, D, lam, V, K_swudi)
    K_aswudi = participation_sqrt_rank(lam.cpu())
    methods["aswudi_sqrt"] = hard_truncation_theta(theta0, D, lam, V, K_aswudi)
    return methods


def part_a_proxy_only(blob: Dict, device: torch.device, max_layers=None):
    layer_names = blob["layer_names"]
    if max_layers is not None:
        layer_names = layer_names[:max_layers]
    rows = []
    for i, name in enumerate(layer_names):
        v = blob["stacks"][name].to(torch.float32).to(device)
        thetas = all_method_thetas(v)
        proxies = {k: wudi_proxy_layer(t, v) for k, t in thetas.items()}
        rows.append({"layer": name, "shape": list(v.shape[1:]),
                     "proxies": proxies})
        if (i + 1) % 16 == 0:
            print(f"  [10.2A] {i+1}/{len(layer_names)}", flush=True)

    # Aggregate per method
    import statistics as st
    method_keys = list(rows[0]["proxies"].keys())
    summary = {
        "n_layers": len(rows),
        "method_total_proxy": {
            k: sum(r["proxies"][k] for r in rows) for k in method_keys
        },
        "method_median_proxy": {
            k: st.median([r["proxies"][k] for r in rows]) for k in method_keys
        },
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Part B: Real-interference subset using forward activations
# ---------------------------------------------------------------------------

def get_subset_layers(blob: Dict, n_subset: int = 3) -> List[str]:
    """Return early/mid/late MLP fc1 layers for activation collection."""
    names = blob["layer_names"]
    fc1 = [n for n in names if "mlp.fc1.weight" in n]
    if not fc1:
        return names[:n_subset]
    L = len(fc1)
    if n_subset >= L:
        return fc1
    idxs = [int(round(i * (L - 1) / (n_subset - 1))) for i in range(n_subset)]
    return [fc1[i] for i in idxs]


DATASET_REPO_OVERRIDES = {
    "mnist": "mnist",
    # all others are tanganke/{task}
}


def get_image_dataset(task_name: str, n_samples: int = 32):
    """Load a small calibration set for one CLIP TA8 task."""
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    repo = DATASET_REPO_OVERRIDES.get(task_name, f"tanganke/{task_name}")
    ds = load_dataset(repo, split="test",
                      cache_dir=str(HF_CACHE.parent / "datasets"))
    if n_samples > 0 and n_samples < len(ds):
        ds = ds.select(range(n_samples))
    return ds


def collect_activations(model, image_tensors: torch.Tensor,
                        layer_names: List[str],
                        device: torch.device) -> Dict[str, torch.Tensor]:
    """Forward image_tensors through model and collect *input* activations
    to each named 2-D layer.
    """
    name_to_module = {n: m for n, m in model.named_modules()
                      if hasattr(m, "weight")}
    # Strip the .weight suffix to get module name
    target = {}
    for n in layer_names:
        if n.endswith(".weight"):
            mod_name = n[:-len(".weight")]
            if mod_name in name_to_module:
                target[mod_name] = n
    captures: Dict[str, List[torch.Tensor]] = {n: [] for n in target.values()}
    handles = []
    def _make_hook(mod_name):
        target_name = target[mod_name]
        def _hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            # CLIP fc1: input is (B, seq, hidden). Flatten to (B*seq, hidden).
            x = x.reshape(-1, x.shape[-1]).detach().cpu().float()
            captures[target_name].append(x)
        return _hook
    for mod_name in target:
        mod = name_to_module[mod_name]
        handles.append(mod.register_forward_hook(_make_hook(mod_name)))
    try:
        model.eval()
        with torch.no_grad():
            for i in range(0, image_tensors.shape[0], 16):
                batch = image_tensors[i:i+16].to(device)
                model(pixel_values=batch)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for n, lst in captures.items():
        if lst:
            out[n] = torch.cat(lst, dim=0)  # (M, d)
    return out


def real_interference(theta: torch.Tensor, v_per_task: List[torch.Tensor],
                      x_per_task: List[torch.Tensor]) -> float:
    """Î(θ) = Σ_i (1/M_i) Σ_m ||(θ - v_i) x_{i,m}||²."""
    total = 0.0
    for v_i, x_i in zip(v_per_task, x_per_task):
        diff = theta - v_i
        proj = x_i @ diff.T            # (M, m)
        total += float((proj * proj).sum(dim=1).mean().item())
    return total


def part_b_real_subset(blob: Dict, device: torch.device,
                       n_samples_per_task: int = 32,
                       tasks_subset: List[str] = None,
                       n_layer_subset: int = 3):
    """Run forward passes on a small image subset to estimate real per-layer
    interference for each merge method."""
    if tasks_subset is None:
        tasks_subset = ["sun397", "mnist", "dtd"]    # diverse: natural / digits / texture
    layer_subset = get_subset_layers(blob, n_subset=n_layer_subset)

    print(f"[10.2B] subset layers: {layer_subset}")
    print(f"[10.2B] tasks: {tasks_subset}, samples/task: {n_samples_per_task}")

    # 1) Load pretrained CLIP vision and image processor
    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained(PRETRAINED_REPO,
                                                   cache_dir=str(HF_CACHE))
    pre_model = load_clip_vision(PRETRAINED_REPO).to(device)

    # 2) Build image tensors per task
    task_images: Dict[str, torch.Tensor] = {}
    task_to_repo: Dict[str, str] = {}
    for repo, name in zip(EXPERT_REPOS, TASK_NAMES):
        if name in tasks_subset:
            task_to_repo[name] = repo
    for name in tasks_subset:
        ds = get_image_dataset(name, n_samples_per_task)
        imgs = [ds[i]["image"].convert("RGB") for i in range(len(ds))]
        proc = processor(images=imgs, return_tensors="pt")
        task_images[name] = proc["pixel_values"].cpu()
        print(f"[10.2B] task {name}: {task_images[name].shape}")

    # 3) Per task, load expert and collect input activations to subset layers
    #    using EACH task's expert (we want x_{i,ℓ} = layer-input under expert i,
    #    matching the §10.1 definition: each task uses its own fine-tuned model).
    activations_per_task: Dict[str, Dict[str, torch.Tensor]] = {}
    for task_name in tasks_subset:
        repo = task_to_repo[task_name]
        print(f"[10.2B] loading expert {repo} for activation collection")
        expert = load_clip_vision(repo).to(device)
        acts = collect_activations(expert, task_images[task_name],
                                   layer_subset, device)
        activations_per_task[task_name] = acts
        del expert
        torch.cuda.empty_cache()
    del pre_model
    torch.cuda.empty_cache()

    # 4) Build θ for each method on the subset layers
    out_per_layer = {}
    method_keys = None
    for name in layer_subset:
        if name not in blob["stacks"]:
            continue
        v = blob["stacks"][name].to(torch.float32).to(device)
        thetas = all_method_thetas(v)
        if method_keys is None:
            method_keys = list(thetas.keys())

        # Map task index in v to task name
        v_per_task = []
        x_per_task = []
        # blob["stacks"] columns are ordered by EXPERT_REPOS, which match TASK_NAMES.
        for task_name in tasks_subset:
            i = TASK_NAMES.index(task_name)
            v_per_task.append(v[i])
            x_per_task.append(activations_per_task[task_name][name].to(device))

        proxies = {k: wudi_proxy_layer(t, v) for k, t in thetas.items()}
        real_intf = {k: real_interference(t, v_per_task, x_per_task)
                     for k, t in thetas.items()}
        out_per_layer[name] = {
            "proxies": proxies,
            "real_interference": real_intf,
            "v_norms_per_task": [float(v_i.norm().item()) for v_i in v_per_task],
            "x_norms_per_task": [float(x_i.norm().item()) for x_i in x_per_task],
        }

    return out_per_layer, method_keys, layer_subset, tasks_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "exp_10_2_proxy_calibration"))
    ap.add_argument("--task_vectors", default=str(DEFAULT_OUT / "task_vectors.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_layers_a", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--n_layer_subset", type=int, default=3)
    ap.add_argument("--skip_part_b", action="store_true")
    args = ap.parse_args()

    configure_torch_for_diagnostics()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.task_vectors, map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Part A
    rows_a, summary_a = part_a_proxy_only(blob, device, max_layers=args.max_layers_a)
    write_json(out_dir / "proxy_only.json", {"rows": rows_a, "summary": summary_a})
    print("[10.2A] summary:", json.dumps(summary_a, indent=2))

    if args.skip_part_b:
        return

    # Part B
    out_b, method_keys, layer_subset, tasks_subset = part_b_real_subset(
        blob, device, n_samples_per_task=args.n_samples,
        n_layer_subset=args.n_layer_subset,
    )

    # Summary: for each layer, do proxy-rank and real-rank agree?
    agreement = []
    for layer_name, payload in out_b.items():
        p_order = sorted(payload["proxies"].items(), key=lambda x: x[1])
        r_order = sorted(payload["real_interference"].items(), key=lambda x: x[1])
        # Check: identical method ordering?
        p_rank = {k: i for i, (k, _) in enumerate(p_order)}
        r_rank = {k: i for i, (k, _) in enumerate(r_order)}
        keys = list(payload["proxies"].keys())
        # Spearman corr
        n = len(keys)
        rs = list(range(n))
        ps = [p_rank[k] for k in keys]
        rrs = [r_rank[k] for k in keys]
        # Pearson on ranks (Spearman)
        mp = sum(ps) / n; mr = sum(rrs) / n
        num = sum((ps[i]-mp)*(rrs[i]-mr) for i in range(n))
        denp = math.sqrt(sum((ps[i]-mp)**2 for i in range(n)))
        denr = math.sqrt(sum((rrs[i]-mr)**2 for i in range(n)))
        rho = num / (denp * denr) if denp > 0 and denr > 0 else float("nan")
        # Pearson of values
        ps_val = [payload["proxies"][k] for k in keys]
        rs_val = [payload["real_interference"][k] for k in keys]
        mpv = sum(ps_val)/n; mrv = sum(rs_val)/n
        nv = sum((ps_val[i]-mpv)*(rs_val[i]-mrv) for i in range(n))
        dpv = math.sqrt(sum((p-mpv)**2 for p in ps_val))
        drv = math.sqrt(sum((r-mrv)**2 for r in rs_val))
        pearson = nv / (dpv*drv) if dpv > 0 and drv > 0 else float("nan")
        agreement.append({
            "layer": layer_name, "spearman_proxy_real": rho,
            "pearson_proxy_real": pearson,
            "best_proxy": p_order[0][0], "best_real": r_order[0][0],
        })

    write_json(out_dir / "real_subset.json", {
        "tasks": tasks_subset,
        "layers": layer_subset,
        "n_samples_per_task": args.n_samples,
        "method_keys": method_keys,
        "per_layer": out_b,
        "agreement": agreement,
    })
    print("[10.2B] per-layer agreement:")
    for a in agreement:
        print(f"  {a['layer']}: spearman={a['spearman_proxy_real']:.3f}, "
              f"pearson={a['pearson_proxy_real']:.3f}, "
              f"best_proxy={a['best_proxy']}, best_real={a['best_real']}")


if __name__ == "__main__":
    main()
