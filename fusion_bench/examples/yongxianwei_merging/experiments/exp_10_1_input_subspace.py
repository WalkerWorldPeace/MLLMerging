"""§10.1 — Input subspace capture experiment.

For each task i ∈ {sun397, mnist, dtd, eurosat} and each subset MLP fc1 layer
ℓ ∈ {early, mid, late}:
    - Collect a calibration sample of input activations x_{i,ℓ} ∈ R^{M × d}
      using task i's fine-tuned expert.
    - Estimate Σ̂_{i,ℓ} = (1/M) X^T X.
    - Compute right-singular subspace of v_{i,ℓ}: V_K from SVD(v_{i,ℓ}).
    - Capture(K) = tr(V_K^T Σ̂ V_K) / tr(Σ̂).
    - Random baseline: average capture over 5 random orthonormal bases of size K.
    - Alignment(C_i, Σ̂_i) = ⟨C_i / ||C_i||, Σ̂_i / ||Σ̂_i||⟩  cosine

Output: capture curves and alignment values that confirm or refute §2.1's
input-subspace assumption.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (DEFAULT_OUT, HF_CACHE, PRETRAINED_REPO, EXPERT_REPOS,
                     TASK_NAMES, load_clip_vision, write_json,
                     configure_torch_for_diagnostics)
from exp_10_2_calibration import get_image_dataset, collect_activations


def cap_curve(V_K: torch.Tensor, Sigma: torch.Tensor) -> float:
    """tr(V_K^T Σ V_K) / tr(Σ)."""
    num = torch.trace(V_K.T @ Sigma @ V_K)
    den = torch.trace(Sigma).clamp(min=1e-30)
    return float((num / den).item())


def random_basis(d: int, K: int, n_repeat: int = 5) -> List[torch.Tensor]:
    bases = []
    for _ in range(n_repeat):
        Q = torch.linalg.qr(torch.randn(d, K)).Q
        bases.append(Q)
    return bases


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal angles between two subspaces, given as orthonormal columns.

    σ_i = singular values of A^T B; angles are arccos(σ_i).
    """
    s = torch.linalg.svdvals(A.T @ B)
    s = s.clamp(min=-1.0, max=1.0)
    return torch.acos(s)


def diagnose_layer_task(name: str, v_i: torch.Tensor,
                        x_i: torch.Tensor) -> Dict:
    """v_i: (m, d) task vector; x_i: (M, d) input activations."""
    m, d = v_i.shape
    M = x_i.shape[0]
    Sigma = (x_i.T @ x_i) / M                            # (d, d)
    # Right-singular subspace of v_i
    U, S, Vt = torch.linalg.svd(v_i.float(), full_matrices=False)
    V = Vt.T                                              # (d, k_v) cols are right-sing vecs
    # Capture curves at K
    K_grid = sorted(set([1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, min(d, 384), min(d, 512), min(d-1, 768-1)] +
                          [int(round(d * f)) for f in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)]))
    K_grid = [k for k in K_grid if 1 <= k <= V.shape[1]]
    cap = []
    cap_random_mean = []
    rng = torch.Generator(device=v_i.device).manual_seed(0)
    for K in K_grid:
        # task-vector subspace
        cap.append(cap_curve(V[:, :K], Sigma))
        # random subspace baseline
        rcs = []
        for _ in range(5):
            Z = torch.randn(d, K, generator=rng, device=v_i.device)
            Q, _ = torch.linalg.qr(Z)
            rcs.append(cap_curve(Q, Sigma))
        cap_random_mean.append(sum(rcs) / len(rcs))

    # Alignment between v_i^T v_i / ||v||² and Σ / tr(Σ)
    A = (v_i.T @ v_i)
    nA = float(A.norm().item()) + 1e-30
    nS = float(Sigma.norm().item()) + 1e-30
    align = float((A * Sigma).sum().item() / (nA * nS))

    # Principal angles: V_top_K_taskvec vs top-K eigenspace of Σ for K=K_align (small)
    K_align = min(64, V.shape[1])
    L, Q = torch.linalg.eigh(Sigma)
    Q_top = Q[:, torch.argsort(L, descending=True)[:K_align]]
    angles = principal_angles(V[:, :K_align], Q_top)
    return {
        "layer": name,
        "shape": [m, d],
        "M": M,
        "K_grid": K_grid,
        "capture_taskvec": cap,
        "capture_random_mean": cap_random_mean,
        "capture_gap_mean": [c - r for c, r in zip(cap, cap_random_mean)],
        "alignment": align,
        "principal_angles_first10_deg": [float(math.degrees(a)) for a in angles[:10].tolist()],
        "trace_sigma": float(torch.trace(Sigma).item()),
        "trace_v_outer": float(torch.trace(v_i.T @ v_i).item()),
    }


def get_subset_layers(blob: Dict, n_subset: int = 3) -> List[str]:
    names = blob["layer_names"]
    fc1 = [n for n in names if "mlp.fc1.weight" in n]
    if not fc1:
        return names[:n_subset]
    L = len(fc1)
    if n_subset >= L:
        return fc1
    idxs = [int(round(i * (L - 1) / (n_subset - 1))) for i in range(n_subset)]
    return [fc1[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "exp_10_1_input_subspace"))
    ap.add_argument("--task_vectors", default=str(DEFAULT_OUT / "task_vectors.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_samples", type=int, default=128,
                    help="Calibration samples per task. With 50 patches/sample, M ~= n_samples * 50.")
    ap.add_argument("--n_layer_subset", type=int, default=3)
    ap.add_argument("--tasks", nargs="+", default=["sun397", "mnist", "dtd", "eurosat"])
    args = ap.parse_args()

    configure_torch_for_diagnostics()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.task_vectors, map_location="cpu", weights_only=False)
    layer_subset = get_subset_layers(blob, n_subset=args.n_layer_subset)
    print(f"[10.1] subset layers: {layer_subset}")
    print(f"[10.1] tasks: {args.tasks}, samples/task: {args.n_samples}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained(PRETRAINED_REPO,
                                                   cache_dir=str(HF_CACHE))

    task_to_repo = dict(zip(TASK_NAMES, EXPERT_REPOS))
    task_images: Dict[str, torch.Tensor] = {}
    for name in args.tasks:
        ds = get_image_dataset(name, args.n_samples)
        imgs = [ds[i]["image"].convert("RGB") for i in range(len(ds))]
        proc = processor(images=imgs, return_tensors="pt")
        task_images[name] = proc["pixel_values"]
        print(f"[10.1] task {name}: {task_images[name].shape}")

    # Collect activations per task using each task's expert
    activations: Dict[str, Dict[str, torch.Tensor]] = {}
    for task_name in args.tasks:
        repo = task_to_repo[task_name]
        print(f"[10.1] collecting activations for {repo}", flush=True)
        expert = load_clip_vision(repo).to(device)
        acts = collect_activations(expert, task_images[task_name], layer_subset, device)
        activations[task_name] = acts
        del expert
        torch.cuda.empty_cache()

    # Diagnose each (task, layer)
    out = []
    for task_name in args.tasks:
        i = TASK_NAMES.index(task_name)
        for layer_name in layer_subset:
            v_i = blob["stacks"][layer_name][i].to(torch.float32).to(device)   # (m, d)
            x_i = activations[task_name][layer_name].to(device)
            print(f"  [10.1] {task_name} @ {layer_name.split('.')[-3]}: "
                  f"v_i={tuple(v_i.shape)}, x_i={tuple(x_i.shape)}", flush=True)
            r = diagnose_layer_task(layer_name, v_i, x_i)
            r["task"] = task_name
            out.append(r)

    # Aggregate: capture gap and alignment heatmap-ready table
    import statistics as st
    capture_at_K_eq_d_quarter = []   # K=d/4 capture - random
    alignments = []
    angle_means = []
    for r in out:
        # Find K closest to d/4
        d = r["shape"][1]
        K_target = d // 4
        idx = min(range(len(r["K_grid"])), key=lambda i: abs(r["K_grid"][i] - K_target))
        capture_at_K_eq_d_quarter.append({
            "task": r["task"], "layer": r["layer"],
            "K": r["K_grid"][idx], "K_ratio": r["K_grid"][idx] / d,
            "capture_taskvec": r["capture_taskvec"][idx],
            "capture_random": r["capture_random_mean"][idx],
            "gap": r["capture_gap_mean"][idx],
        })
        alignments.append(r["alignment"])
        angle_means.append(sum(r["principal_angles_first10_deg"]) / 10)

    summary = {
        "tasks": args.tasks,
        "layers": layer_subset,
        "n_samples_per_task": args.n_samples,
        "alignment_mean": st.fmean(alignments),
        "alignment_median": st.median(alignments),
        "alignment_min": min(alignments),
        "alignment_max": max(alignments),
        "first10_principal_angle_deg_mean": st.fmean(angle_means),
        "first10_principal_angle_deg_median": st.median(angle_means),
        "capture_gap_at_d_over_4_mean": st.fmean(c["gap"] for c in capture_at_K_eq_d_quarter),
        "capture_gap_at_d_over_4_median": st.median(c["gap"] for c in capture_at_K_eq_d_quarter),
    }
    write_json(out_dir / "per_pair.json", out)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "capture_at_d_over_4.json", capture_at_K_eq_d_quarter)
    print("[10.1] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
