"""§10.3 — Optimizer trajectory diagnostic.

For one representative MLP fc1 layer of the CLIP-ViT-B/32 vision encoder,
we run pure SGD and Adam on the WUDI quadratic
    L(θ) = Σ_i ||(θ - v_i) v_i^T||² / ||v_i||²
saving θ_n at a logarithmic schedule. For each saved θ_n we compute the
empirical filter coefficients
    ĥ_{k,n} = ⟨(θ_n - θ_0) q_k, B q_k / λ_k⟩ / ||B q_k / λ_k||²
where B = D - θ_0 C and (λ_k, q_k) are eigenpairs of C.

Theory predictions (§4):
    - GD with step η ∈ (0, 2/λ_max): ĥ_{k,n} = 1 - (1 - η λ_k)^n   (Landweber)
    - Gradient flow at time t:        ĥ_k(t) = 1 - exp(-t λ_k)
    - Adam: no exact filter form, but should empirically saturate large-λ
      directions before small-λ ones.

We fit per-step the best Landweber n_eff (for SGD) and exponential time t_eff
(for Adam), and report R^2 of the fit + Spearman corr of ĥ vs λ at each step.

Output:
    outputs/yongxianwei_merging/theory_diagnostics/exp_10_3_optimizer/
        per_layer_<layer>.json    (per-step ĥ_k, fit params, fit R²)
        summary.json
        figures/                  (filter curve, fit-quality, accuracy step-sweep)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (DEFAULT_OUT, compute_C_D, eigh_descending, write_json,
                     configure_torch_for_diagnostics)


def empirical_h(theta_n: torch.Tensor, theta0: torch.Tensor,
                B: torch.Tensor, V: torch.Tensor, lam: torch.Tensor,
                eps: float = 1e-12) -> torch.Tensor:
    """Per-direction empirical filter coefficient for a saved θ_n.

    Recipe
    ------
    Define R_n = θ_n - θ_0 (m × d). The "noise-free" full inverse residual is
    B q_k / λ_k. The fitted coefficient h_k is the scalar projection of
    R_n q_k onto B q_k / λ_k. This generalises the scalar 1-D residual model
    R_{n,k} = h_{k,n} · (B q_k / λ_k).
    """
    R_n = theta_n - theta0                              # (m, d)
    Bq = B @ V                                          # (m, d)
    target = Bq / lam.clamp(min=eps).unsqueeze(0)       # (m, d) where each col is B q_k / λ_k
    Rq = R_n @ V                                        # (m, d) where each col is R_n q_k
    num = (Rq * target).sum(dim=0)                      # (d,)
    den = (target * target).sum(dim=0).clamp(min=eps)   # (d,)
    return num / den


def landweber_h(eta: float, lam: torch.Tensor, n: int) -> torch.Tensor:
    factor = (1.0 - eta * lam.clamp(min=0)).clamp(min=-1.0, max=1.0)
    return 1.0 - factor.pow(n)


def exponential_h(t: float, lam: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-t * lam.clamp(min=0))


def fit_t_exp(h: torch.Tensor, lam: torch.Tensor) -> float:
    """Fit h_k ≈ 1 - exp(-t λ_k) by 1-D root-find on residual sum.

    Using log on negative residuals: -ln(1-h) ≈ t λ. Robust to outliers via
    Huber-ish weighting (drop saturated and clearly-negative h).
    """
    mask = (h > 0.05) & (h < 0.95) & (lam > 0)
    if mask.sum() < 3:
        return float("nan")
    one_m_h = (1.0 - h[mask]).clamp(min=1e-6)
    rhs = -torch.log(one_m_h)        # ≈ t λ
    lhs = lam[mask]
    # least-squares slope through origin
    t = float((rhs * lhs).sum() / (lhs * lhs).sum().clamp(min=1e-30))
    return t


def fit_n_landweber(h: torch.Tensor, lam: torch.Tensor, eta: float) -> float:
    """Fit h_k ≈ 1 - (1 - η λ_k)^n by linearising:
        log(1 - h_k) = n log(1 - η λ_k)
    """
    mask = (h > 0.05) & (h < 0.95) & (lam > 0) & (1 - eta * lam > 0)
    if mask.sum() < 3:
        return float("nan")
    one_m_h = (1.0 - h[mask]).clamp(min=1e-6)
    # avoid log(<=0)
    base = (1.0 - eta * lam[mask]).clamp(min=1e-6)
    rhs = torch.log(one_m_h)         # = n log(base) ; both negative
    lhs = torch.log(base)
    n = float((rhs * lhs).sum() / (lhs * lhs).sum().clamp(min=1e-30))
    return n


def fit_quality(h_emp: torch.Tensor, h_pred: torch.Tensor) -> Dict[str, float]:
    """R^2 (1 - SSres/SStot) and Spearman rank correlation."""
    r = h_emp - h_pred
    sstot = ((h_emp - h_emp.mean()) ** 2).sum().clamp(min=1e-30)
    ssres = (r * r).sum()
    r2 = float(1.0 - ssres / sstot)
    # Spearman: rank correlation of h_emp vs h_pred
    def _rank(x):
        return x.argsort().argsort().float()
    rho = float(torch.corrcoef(torch.stack([_rank(h_emp), _rank(h_pred)]))[0, 1])
    return {"r2": r2, "spearman": rho}


def run_optimizer(C: torch.Tensor, D: torch.Tensor, theta0: torch.Tensor,
                  v_stack: torch.Tensor, l2_sq: torch.Tensor,
                  optimizer: str, n_steps: int, lr: float,
                  device: torch.device,
                  log_steps: List[int]) -> Dict[int, torch.Tensor]:
    """Run optimizer on the WUDI quadratic, save θ at log_steps.

    The exact gradient is 2 (θ C - D), so we can avoid materialising tvs again.
    But we want Adam's behaviour to match the original WUDI formulation, so we
    write the loss explicitly and let autograd handle it.
    """
    theta = theta0.detach().clone().to(device).requires_grad_(True)
    v_stack = v_stack.to(device)
    inv_norms_unsq = (1.0 / l2_sq.to(device).clamp(min=1e-12)).view(-1, 1, 1)

    if optimizer == "sgd":
        opt = torch.optim.SGD([theta], lr=lr, momentum=0.0)
    elif optimizer == "adam":
        opt = torch.optim.Adam([theta], lr=lr, weight_decay=0)
    else:
        raise ValueError(optimizer)

    saved = {}
    if 0 in log_steps:
        saved[0] = theta.detach().cpu().clone()
    log_set = set(log_steps)

    for step in range(1, n_steps + 1):
        opt.zero_grad()
        disturbing = theta.unsqueeze(0) - v_stack       # (N, m, d)
        product = torch.matmul(disturbing, v_stack.transpose(1, 2))  # (N, m, m)
        loss = (product.square() * inv_norms_unsq).sum()
        loss.backward()
        opt.step()
        if step in log_set:
            saved[step] = theta.detach().cpu().clone()
    return saved


def pick_layers(stack_path: Path, max_layers: int = 4) -> List[str]:
    """Pick a small representative set of layer names: one early/mid/late MLP fc1."""
    blob = torch.load(stack_path, map_location="cpu", weights_only=False)
    names = blob["layer_names"]
    # Filter to MLP fc1 layers (3072 × 768)
    mlp_fc1 = [n for n in names if "mlp.fc1.weight" in n]
    if not mlp_fc1:
        mlp_fc1 = names
    # Pick early / mid / late (and one extra)
    L = len(mlp_fc1)
    if max_layers >= L:
        return mlp_fc1
    idxs = [int(round(i * (L - 1) / (max_layers - 1))) for i in range(max_layers)]
    return [mlp_fc1[i] for i in idxs]


def diagnose_layer(name: str, v_stack: torch.Tensor, device: torch.device,
                   n_steps: int, log_steps: List[int]) -> Dict:
    v_stack = v_stack.to(torch.float32)
    C, D, l2_sq, theta0 = compute_C_D(v_stack)
    C = C.to(device); D = D.to(device); theta0 = theta0.to(device)
    lam, V = eigh_descending(C)
    B = D - theta0 @ C
    lam_max = float(lam[0].item())
    # SGD step size: η · λ_max ≈ 0.5 → step length ~ half of stability boundary.
    eta = 0.5 / lam_max if lam_max > 0 else 1e-3

    out = {
        "layer": name,
        "shape": list(v_stack.shape),
        "lam_max": lam_max,
        "lam_median": float(lam.median().item()),
        "n_steps": n_steps,
        "log_steps": list(log_steps),
        "lr_sgd": eta,
        "lr_adam": 1e-5,
        "spectra": {
            "lam": lam.cpu().tolist(),
        },
        "trajectories": {},
    }

    for opt_name, lr in [("sgd", eta), ("adam", 1e-5)]:
        saved = run_optimizer(C.cpu(), D.cpu(), theta0.cpu(), v_stack, l2_sq,
                              optimizer=opt_name, n_steps=n_steps, lr=lr,
                              device=device, log_steps=log_steps)
        traj = {"steps": [], "h_emp": [], "fit": []}
        # Move B, V, lam to cpu for h_emp computation (small tensors)
        B_cpu = B.cpu(); V_cpu = V.cpu(); lam_cpu = lam.cpu(); theta0_cpu = theta0.cpu()
        for step, theta_n in saved.items():
            h_emp = empirical_h(theta_n, theta0_cpu, B_cpu, V_cpu, lam_cpu)
            traj["steps"].append(int(step))
            # Down-sample h_emp to top-200 dims for storage
            keep = min(256, lam_cpu.numel())
            traj["h_emp"].append(h_emp[:keep].tolist())
            # Fit theory curves for this step
            if opt_name == "sgd":
                n_lan = fit_n_landweber(h_emp, lam_cpu, eta)
                h_pred = landweber_h(eta, lam_cpu, n_lan if not math.isnan(n_lan) else 0)
                fit = {"type": "landweber", "n_eff": n_lan,
                       **fit_quality(h_emp, h_pred)}
            else:
                t_eff = fit_t_exp(h_emp, lam_cpu)
                h_pred = exponential_h(t_eff if not math.isnan(t_eff) else 0, lam_cpu)
                fit = {"type": "exponential", "t_eff": t_eff,
                       **fit_quality(h_emp, h_pred)}
            traj["fit"].append(fit)
        out["trajectories"][opt_name] = traj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT / "exp_10_3_optimizer"))
    ap.add_argument("--task_vectors", default=str(DEFAULT_OUT / "task_vectors.pt"))
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--max_layers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    configure_torch_for_diagnostics()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.task_vectors, map_location="cpu", weights_only=False)
    layer_names = pick_layers(Path(args.task_vectors), max_layers=args.max_layers)
    print(f"[10.3] diagnosing layers: {layer_names}")

    log_steps = sorted(set([0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 250, args.n_steps]))
    log_steps = [s for s in log_steps if s <= args.n_steps]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    summary = {
        "layers": layer_names,
        "n_steps": args.n_steps,
        "log_steps": log_steps,
        "device": str(device),
    }

    per_layer = {}
    for name in layer_names:
        if name not in blob["stacks"]:
            print(f"[10.3] skipping {name} (not in cache)")
            continue
        v_stack = blob["stacks"][name].to(torch.float32)
        t0 = time.time()
        result = diagnose_layer(name, v_stack, device, args.n_steps, log_steps)
        per_layer[name] = result
        # log compact summary
        sgd_fits = [f["r2"] for f in result["trajectories"]["sgd"]["fit"]]
        adam_fits = [f["r2"] for f in result["trajectories"]["adam"]["fit"]]
        print(f"  [10.3] {name}: SGD R² last {sgd_fits[-1]:.3f}, "
              f"Adam R² last {adam_fits[-1]:.3f}, "
              f"({time.time()-t0:.1f}s)", flush=True)

    write_json(out_dir / "per_layer.json", per_layer)
    write_json(out_dir / "summary.json", {
        **summary,
        "fit_quality_summary": {
            "sgd_landweber_r2_final": [
                per_layer[n]["trajectories"]["sgd"]["fit"][-1]["r2"]
                for n in per_layer
            ],
            "adam_exponential_r2_final": [
                per_layer[n]["trajectories"]["adam"]["fit"][-1]["r2"]
                for n in per_layer
            ],
        },
    })


if __name__ == "__main__":
    main()
