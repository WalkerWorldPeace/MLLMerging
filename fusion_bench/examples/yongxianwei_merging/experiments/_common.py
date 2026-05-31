"""Shared utilities for §10 diagnostic experiments.

This module is import-only — running it as a script does nothing. The diagnostic
scripts (`exp_*.py`) build on these primitives.

Conventions
-----------
- All tensors live in float32 on CPU until pushed to a device explicitly.
- Per-layer 2-D task-vector stacks are shaped (N_experts, m, d) where m=output,
  d=input. We compute C = Σ_i (v_i^T v_i) / ||v_i||_F^2 ∈ R^{d×d}.
- D = Σ_i v_i A_i = Σ_i (v_i v_i^T v_i) / ||v_i||_F^2  ∈ R^{m×d}.
- Eigendecomposition is descending: λ_1 ≥ ... ≥ λ_d, with eigenvectors columns
  of V (d × d).
- All `_solve_*` helpers operate per-layer and return (m, d) merged matrices.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[3])))
HF_CACHE = Path(os.environ.get("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))) / "hub"
EXPERT_REPOS = [
    "tanganke/clip-vit-base-patch32_sun397",
    "tanganke/clip-vit-base-patch32_stanford-cars",
    "tanganke/clip-vit-base-patch32_resisc45",
    "tanganke/clip-vit-base-patch32_eurosat",
    "tanganke/clip-vit-base-patch32_svhn",
    "tanganke/clip-vit-base-patch32_gtsrb",
    "tanganke/clip-vit-base-patch32_mnist",
    "tanganke/clip-vit-base-patch32_dtd",
]
TASK_NAMES = [
    "sun397", "stanford-cars", "resisc45", "eurosat",
    "svhn", "gtsrb", "mnist", "dtd",
]
PRETRAINED_REPO = "openai/clip-vit-base-patch32"

DEFAULT_OUT = REPO_ROOT / "outputs" / "yongxianwei_merging" / "theory_diagnostics"
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_clip_vision(repo_id: str):
    """Return CLIPVisionModel from local HF hub cache."""
    from transformers import CLIPVisionModel
    return CLIPVisionModel.from_pretrained(
        repo_id, cache_dir=str(HF_CACHE),
    )


def load_clip_full(repo_id: str = PRETRAINED_REPO):
    """Load the full CLIPModel (used to get text projection & logit_scale for eval)."""
    from transformers import CLIPModel
    return CLIPModel.from_pretrained(repo_id, cache_dir=str(HF_CACHE))


def get_2d_layer_names(model, max_layers: Optional[int] = None) -> List[str]:
    """Return mergeable 2-D layer names (excludes 1-D embeddings, lm_head, etc)."""
    names = [
        n for n, p in model.named_parameters()
        if p.ndim == 2 and "lm_head" not in n and "embeddings" not in n
    ]
    if max_layers is not None:
        names = names[:max_layers]
    return names


def collect_task_vectors(
    pretrained_state: Dict[str, torch.Tensor],
    expert_states: List[Dict[str, torch.Tensor]],
    layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """Return name -> stacked (N, m, d) float32 task-vector tensor."""
    out: Dict[str, torch.Tensor] = {}
    for name in layer_names:
        if name not in pretrained_state:
            continue
        base = pretrained_state[name].to(torch.float32)
        deltas = []
        for ex in expert_states:
            if name not in ex or ex[name].shape != base.shape:
                deltas = None
                break
            deltas.append(ex[name].to(torch.float32) - base)
        if deltas is None or len(deltas) < 2:
            continue
        out[name] = torch.stack(deltas, dim=0)
    return out


def load_pool_state_dicts() -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Load _pretrained_ + 8 expert vision-model state dicts.

    The pretrained CLIPVisionModel and tanganke's expert CLIPVisionModel share
    parameter names (vision_model.encoder.layers.*.{...}.weight), so we just
    load each as CLIPVisionModel and grab its state dict.
    """
    print("[load] loading pretrained CLIPVisionModel openai/clip-vit-base-patch32", flush=True)
    pre = load_clip_vision(PRETRAINED_REPO)
    pre_sd = {n: p.detach().clone() for n, p in pre.state_dict().items()}
    del pre
    expert_sds = []
    for repo in EXPERT_REPOS:
        print(f"[load] expert {repo}", flush=True)
        m = load_clip_vision(repo)
        expert_sds.append({n: p.detach().clone() for n, p in m.state_dict().items()})
        del m
    return pre_sd, expert_sds


# ---------------------------------------------------------------------------
# Spectral / WUDI primitives (per-layer)
# ---------------------------------------------------------------------------

def compute_C_D(v_stack: torch.Tensor, eps: float = 1e-12):
    """Compute (C, D, l2_sq) from (N, m, d) task-vector stack.

    Returns
    -------
    C   : (d, d) — Σ_i v_i^T v_i / ||v_i||_F^2
    D   : (m, d) — Σ_i v_i (v_i^T v_i) / ||v_i||_F^2
    l2  : (N,)   — ||v_i||_F^2 (clamped)
    theta0 : (m, d) — Σ_i v_i (initial WUDI sum)
    """
    N, m, d = v_stack.shape
    flat = v_stack.reshape(N, -1)
    l2_sq = (flat * flat).sum(dim=-1).clamp(min=eps)        # (N,)
    inv = 1.0 / l2_sq                                      # (N,)
    # C
    Bs = torch.einsum("nab,nac->nbc", v_stack, v_stack)    # (N, d, d)
    C = torch.einsum("n,nbc->bc", inv, Bs)
    # D = sum_i v_i @ (v_i^T v_i) / ||v_i||^2 = sum_i (v_i @ B_i) * inv
    D = torch.einsum("n,nab,nbc->ac", inv, v_stack, Bs)
    theta0 = v_stack.sum(dim=0)
    return C, D, l2_sq, theta0


def eigh_descending(C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (lambda, V) sorted descending. C is symmetric PSD."""
    w, V = torch.linalg.eigh(C)             # ascending
    w = w.clamp(min=0.0)
    idx = torch.argsort(w, descending=True)
    return w[idx], V[:, idx]


def filter_to_theta(theta0: torch.Tensor, D: torch.Tensor,
                    lam: torch.Tensor, V: torch.Tensor,
                    h: torch.Tensor) -> torch.Tensor:
    """Apply spectral filter h (length d) to produce merged θ.

    From §3 of theory_framework.md:
        θ̂ = θ_0 - θ_0 V diag(h) V^T + D V diag(h/λ) V^T
    """
    eps = 1e-30
    h_over_lam = h / lam.clamp(min=eps)
    # Mask any λ that is zero -> their h must already be zero, but force-safe.
    h_over_lam = torch.where(lam > 0, h_over_lam, torch.zeros_like(h_over_lam))
    Vh = V * h.unsqueeze(0)
    VhL = V * h_over_lam.unsqueeze(0)
    theta = theta0 - theta0 @ Vh @ V.T + D @ VhL @ V.T
    return theta


def closed_form_theta(theta0: torch.Tensor, D: torch.Tensor,
                      lam: torch.Tensor, V: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """θ_cf = D C^{-1} via the eigenbasis (clip tiny λ)."""
    h = torch.where(lam > eps, torch.ones_like(lam), torch.zeros_like(lam))
    return filter_to_theta(theta0, D, lam, V, h)


def hard_truncation_theta(theta0: torch.Tensor, D: torch.Tensor,
                          lam: torch.Tensor, V: torch.Tensor,
                          K: int) -> torch.Tensor:
    """ASWUDI-style hard-K truncation: h_k = 1 for top-K, 0 otherwise."""
    h = torch.zeros_like(lam)
    K = max(0, min(K, lam.numel()))
    if K > 0:
        h[:K] = 1.0
    return filter_to_theta(theta0, D, lam, V, h)


def exponential_filter_theta(theta0: torch.Tensor, D: torch.Tensor,
                             lam: torch.Tensor, V: torch.Tensor,
                             t: float) -> torch.Tensor:
    """IWUDI-style soft filter h_k = 1 - exp(-λ_k t)."""
    h = 1.0 - torch.exp(-t * lam.clamp(min=0))
    return filter_to_theta(theta0, D, lam, V, h)


def landweber_filter_theta(theta0: torch.Tensor, D: torch.Tensor,
                           lam: torch.Tensor, V: torch.Tensor,
                           eta: float, n: int) -> torch.Tensor:
    """GD-style filter h_k = 1 - (1 - eta λ_k)^n.

    Caller must ensure 0 < eta < 2/lam_max.
    """
    factor = (1.0 - eta * lam.clamp(min=0)).clamp(min=-1.0, max=1.0)
    h = 1.0 - factor.pow(n)
    return filter_to_theta(theta0, D, lam, V, h)


# ---------------------------------------------------------------------------
# Rank rules
# ---------------------------------------------------------------------------

def participation_sqrt_rank(lam: torch.Tensor) -> int:
    """K_{√λ} = ⌈ (Σ √λ)^2 / Σ λ ⌉."""
    s = lam.clamp(min=0).sqrt()
    num = s.sum().pow(2)
    den = lam.clamp(min=0).sum().clamp(min=1e-30)
    return int(math.ceil(float(num / den)))


def participation_rank(lam: torch.Tensor) -> int:
    """K_λ = ⌈ (Σ λ)^2 / Σ λ^2 ⌉."""
    a = lam.clamp(min=0)
    num = a.sum().pow(2)
    den = (a * a).sum().clamp(min=1e-30)
    return int(math.ceil(float(num / den)))


def gavish_donoho_rank(lam: torch.Tensor, m: int, d: int) -> int:
    """Gavish-Donoho 2014 optimal hard threshold under unknown noise.

    s_k = √λ_k. Threshold τ = ω(β) · median(s) where β = min(m,d)/max(m,d).
    ω(β) = 0.56 β^3 - 0.95 β^2 + 1.82 β + 1.43  (eq. (5) of GD 2014).
    """
    s = lam.clamp(min=0).sqrt()
    s_med = s.median().item()
    if s_med <= 0:
        return 0
    beta = min(m, d) / max(m, d)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = omega * s_med
    return int((s > tau).sum().item())


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"[write] {path}", flush=True)


def configure_torch_for_diagnostics():
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False  # full precision for spectra
