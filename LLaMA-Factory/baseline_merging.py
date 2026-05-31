"""Standalone implementations of paper baseline merging methods.

Used by ``run_merge_baseline.py`` (Qwen2-VL) and
``run_merge_baseline_internvl.py`` (InternVL2_5) to re-merge the 8 paper
baselines listed in TODO.md §0:
    weight_average, task_arithmetic, ties, dare_ta, dare_ties, svd, iso, wudi.

This module is the cleaned-up, side-effect-free counterpart of
``MLLMerging/LLaMA-Factory/model_merging.py`` and
``MLLMerging/InternVL/internvl_chat/model_merging.py`` — those files run
example generation at import time, so they cannot be reused as-is.

Two corrections vs. the original files:
1. ``svd_merging`` previously gated TSV on ``param_name == 'lm_head.weight'``,
   which is never reachable because ``lm_head`` is in ``exclude_*``. Every 2-D
   weight therefore fell through to a plain delta average and the method
   degenerated to ``weight_average``. Fixed: TSV runs on every 2-D parameter
   in ``param_names_to_merge``.
2. ``LLaMA-Factory/model_merging.py:711`` (dare_ties branch) passed the last
   loop variable ``new_model_to_merge`` (a single model) to ``ties_merging``
   instead of the list ``new_models_to_merge``. The dispatcher in this module
   passes the full list.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm


# ----------------------------------------------------------------------------- #
#  Parameter selection / TaskVector primitives                                  #
# ----------------------------------------------------------------------------- #

def get_param_names_to_merge(input_param_names: Sequence[str],
                             exclude_param_names_regex: Sequence[str]) -> List[str]:
    out = []
    for name in input_param_names:
        if not any(re.match(pat, name) for pat in exclude_param_names_regex):
            out.append(name)
    return out


class TaskVector:
    """delta = finetuned - pretrained over the to-merge subset of parameters."""

    def __init__(self,
                 pretrained_model: Optional[nn.Module] = None,
                 finetuned_model: Optional[nn.Module] = None,
                 exclude_param_names_regex: Optional[Sequence[str]] = None,
                 task_vector_param_dict: Optional[dict] = None):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
            return
        assert pretrained_model is not None and finetuned_model is not None
        pre = {n: p for n, p in pretrained_model.named_parameters()}
        fine = {n: p for n, p in finetuned_model.named_parameters()}
        names = get_param_names_to_merge(list(pre.keys()), exclude_param_names_regex or [])
        self.task_vector_param_dict = {}
        with torch.no_grad():
            for n in names:
                self.task_vector_param_dict[n] = fine[n] - pre[n]

    def __add__(self, other: "TaskVector") -> "TaskVector":
        new = {}
        with torch.no_grad():
            for n, v in self.task_vector_param_dict.items():
                new[n] = v + other.task_vector_param_dict[n]
        return TaskVector(task_vector_param_dict=new)

    def __radd__(self, other):
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module,
                                      scaling_coefficient: float = 1.0) -> dict:
        pre = {n: p for n, p in pretrained_model.named_parameters()}
        with torch.no_grad():
            merged = {n: pre[n] + scaling_coefficient * v
                      for n, v in self.task_vector_param_dict.items()}
        return merged


def copy_params_to_model(params: dict, model: nn.Module) -> None:
    for name, value in model.named_parameters():
        if name in params:
            value.data.copy_(params[name])


def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float,
                              use_rescale: bool, mask_strategy: str) -> torch.Tensor:
    assert 0.0 <= mask_rate <= 1.0
    original_dtype = input_tensor.dtype
    x = input_tensor.float()
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(x, mask_rate)).to(x.device)
        out = x * (1 - mask)
    elif mask_strategy == "magnitude":
        flat = x.flatten()
        k = int(len(flat) * mask_rate)
        kth, _ = flat.abs().kthvalue(k=k, dim=0, keepdim=True)
        keep = (flat.abs() > kth).reshape(x.shape)
        out = x * keep
    else:
        raise ValueError(f"unknown mask_strategy {mask_strategy!r}")
    if use_rescale and mask_rate != 1.0:
        out = out / (1 - mask_rate)
    return out.to(original_dtype)


def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module,
                       exclude_param_names_regex: Sequence[str], weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool,
                       mask_strategy: str) -> dict:
    assert weight_format == "delta_weight", "baseline DARE always uses delta_weight"
    tv = TaskVector(pretrained_model=pretrained_model,
                    finetuned_model=finetuned_model,
                    exclude_param_names_regex=exclude_param_names_regex)
    masked = {}
    with torch.no_grad():
        for name, value in tqdm(tv.task_vector_param_dict.items(), desc="DARE mask"):
            masked[name] = mask_input_with_mask_rate(
                input_tensor=value, mask_rate=weight_mask_rate,
                use_rescale=use_weight_rescale, mask_strategy=mask_strategy,
            )
    masked_tv = TaskVector(task_vector_param_dict=masked)
    return masked_tv.combine_with_pretrained_model(pretrained_model=pretrained_model,
                                                   scaling_coefficient=1.0)


# ----------------------------------------------------------------------------- #
#  Baseline merging methods                                                     #
# ----------------------------------------------------------------------------- #

def weight_average_merging(base_model: nn.Module, models_to_merge: List[nn.Module],
                           exclude_param_names_regex: Sequence[str]) -> dict:
    """Mean of expert weights on the to-merge subset.

    Excluded params are not returned, so the caller leaves them at base
    (matches the convention of every other method in this module).
    """
    base = {n: p for n, p in base_model.named_parameters()}
    names = get_param_names_to_merge(list(base.keys()), exclude_param_names_regex)
    merged = {}
    n_experts = len(models_to_merge)
    expert_states = [m.state_dict() for m in models_to_merge]
    with torch.no_grad():
        for name in tqdm(names, desc="Weight average"):
            acc = expert_states[0][name].clone().to(torch.float32)
            for s in expert_states[1:]:
                acc.add_(s[name].to(torch.float32))
            acc.div_(n_experts)
            merged[name] = acc.to(expert_states[0][name].dtype)
    return merged


def task_arithmetic(base_model: nn.Module, models_to_merge: List[nn.Module],
                    exclude_param_names_regex: Sequence[str],
                    scaling_coefficient: float = 1.0) -> dict:
    tvs = [TaskVector(pretrained_model=base_model, finetuned_model=m,
                      exclude_param_names_regex=exclude_param_names_regex)
           for m in models_to_merge]
    with torch.no_grad():
        merged_tv = tvs[0]
        for tv in tvs[1:]:
            merged_tv = merged_tv + tv
        return merged_tv.combine_with_pretrained_model(
            pretrained_model=base_model, scaling_coefficient=scaling_coefficient,
        )


def ties_merging(base_model: nn.Module, models_to_merge: List[nn.Module],
                 exclude_param_names_regex: Sequence[str],
                 param_value_mask_rate: float = 0.8,
                 scaling_coefficient: float = 1.0) -> dict:
    """TIES merging: magnitude mask + sign election + disjoint mean (per-param)."""

    def mask_smallest(t: torch.Tensor, rate: float) -> torch.Tensor:
        original_dtype = t.dtype
        x = t.float()
        k = int(x.numel() * rate)
        if k == 0:
            return t
        kth = x.reshape(-1).abs().kthvalue(k=k).values
        return (x * (x.abs() >= kth)).to(original_dtype)

    def signs_of(tensors: List[torch.Tensor]) -> torch.Tensor:
        s = torch.sign(sum(tensors))
        if (s == 0).any():
            majority = torch.sign(s.sum())
            s[s == 0] = majority
        return s

    def disjoint_merge(tensors: List[torch.Tensor], signs: torch.Tensor) -> torch.Tensor:
        kept = []
        for t in tensors:
            keep = ((signs > 0) & (t > 0)) | ((signs < 0) & (t < 0))
            kept.append(t * keep)
        denom = sum((p != 0).float() for p in kept)
        return sum(kept) / torch.clamp(denom, min=1.0)

    base = {n: p for n, p in base_model.named_parameters()}
    names = get_param_names_to_merge(list(base.keys()), exclude_param_names_regex)
    expert_states = [m.state_dict() for m in models_to_merge]

    merged = {}
    with torch.no_grad():
        for name in tqdm(names, desc="TIES merge"):
            deltas = [s[name] - base[name] for s in expert_states]
            masked = [mask_smallest(d, param_value_mask_rate) for d in deltas]
            signs = signs_of(masked)
            merged_delta = disjoint_merge(masked, signs)
            merged[name] = base[name] + scaling_coefficient * merged_delta
    return merged


def svd_merging(base_model: nn.Module, models_to_merge: List[nn.Module],
                exclude_param_names_regex: Sequence[str],
                scaling_coefficient: float = 1.0,
                device: str = "cuda") -> dict:
    """TSV-style SVD merge (Wang et al.) — runs on every 2-D parameter.

    Per task vector, take the top ``rank/n_experts`` directions; concat across
    experts; do a second SVD on stacked U / V; rebuild the merged delta with
    the original (concatenated) singular values — see ``svd_merging`` in
    ``MLLMerging/LLaMA-Factory/model_merging.py`` for the original formulation.
    Non-2D parameters are merged by simple delta averaging.
    """
    dev = torch.device(device)
    base = {n: p for n, p in base_model.named_parameters()}
    names = get_param_names_to_merge(list(base.keys()), exclude_param_names_regex)
    expert_states = [m.state_dict() for m in models_to_merge]
    n = len(models_to_merge)
    sv_reduction = 1.0 / n

    merged_delta = {}
    with torch.no_grad():
        for name in tqdm(names, desc="TSV merge"):
            deltas = [s[name] - base[name] for s in expert_states]
            shape = deltas[0].shape
            original_dtype = deltas[0].dtype

            if len(shape) == 2:
                torch.cuda.empty_cache()
                sum_u = sum_s = sum_v = None
                for i, d in enumerate(deltas):
                    vec = d.to(dev).float()
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    r = int(s.shape[0] * sv_reduction)
                    if i == 0:
                        sum_u = torch.zeros_like(u, device=dev)
                        sum_s = torch.zeros_like(s, device=dev)
                        sum_v = torch.zeros_like(v, device=dev)
                    sum_u[:, i * r: (i + 1) * r] = u[:, :r]
                    sum_s[i * r: (i + 1) * r] = s[:r]
                    sum_v[i * r: (i + 1) * r, :] = v[:r, :]
                u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                merged = torch.linalg.multi_dot(
                    [u_u, v_u, torch.diag(sum_s), u_v, v_v]
                ).to(original_dtype).cpu()
                merged_delta[name] = merged
            else:
                acc = deltas[0].clone()
                for i, d in enumerate(deltas[1:], 1):
                    acc.add_((d - acc) / (i + 1))
                merged_delta[name] = acc

    tv = TaskVector(task_vector_param_dict=merged_delta)
    return tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=scaling_coefficient,
    )


def iso_merging(base_model: nn.Module, models_to_merge: List[nn.Module],
                exclude_param_names_regex: Sequence[str],
                scaling_coefficient: float = 1.0,
                device: str = "cuda") -> dict:
    """Iso-C: SVD on the summed delta, replace singular values by their mean."""
    dev = torch.device(device)
    tvs = [TaskVector(pretrained_model=base_model, finetuned_model=m,
                      exclude_param_names_regex=exclude_param_names_regex)
           for m in models_to_merge]
    with torch.no_grad():
        summed = tvs[0]
        for tv in tvs[1:]:
            summed = summed + tv

    merged_delta = {}
    for name, value in tqdm(summed.task_vector_param_dict.items(), desc="Iso-C merge"):
        original_dtype = value.dtype
        if value.dim() == 2:
            x = value.to(dev).float()
            u, s, v = torch.linalg.svd(x, full_matrices=False)
            avg = torch.full_like(s, torch.mean(s))
            merged = torch.linalg.multi_dot([u, torch.diag(avg), v]).to(original_dtype).cpu()
            merged_delta[name] = merged
        else:
            merged_delta[name] = value
    tv = TaskVector(task_vector_param_dict=merged_delta)
    return tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=scaling_coefficient,
    )


def wudi_merging(base_model: nn.Module, models_to_merge: List[nn.Module],
                 exclude_param_names_regex: Sequence[str],
                 scaling_coefficient: float = 1.0,
                 iter_num: int = 300, lr: float = 1e-5,
                 num_chunks: int = 2,
                 device: str = "cuda") -> dict:
    """WUDI v1 — chunked Adam optimisation of a merging vector that minimises
    interference projected onto each expert's column space.

    Exact mirror of ``wudi_merging`` in
    ``MLLMerging/LLaMA-Factory/model_merging.py:449`` (paper baseline);
    included here so the runner does not import the side-effecting script.
    """
    dev = torch.device(device)
    tvs = [TaskVector(pretrained_model=base_model, finetuned_model=m,
                      exclude_param_names_regex=exclude_param_names_regex)
           for m in models_to_merge]

    def optimise(name: str, vectors: torch.Tensor) -> torch.Tensor:
        original_dtype = vectors.dtype
        vectors = vectors.float().to(dev)
        model_num = vectors.shape[0]
        models_per_chunk = (model_num + num_chunks - 1) // num_chunks
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        opt = torch.optim.Adam([merging_vector], lr=lr)
        l2_norms_sq = torch.square(torch.norm(vectors.reshape(model_num, -1), p=2, dim=-1))
        for step in tqdm(range(iter_num), desc=f"WUDI {name}", leave=False):
            opt.zero_grad()
            total_loss = 0.0
            for c in range(num_chunks):
                a = c * models_per_chunk
                b = min((c + 1) * models_per_chunk, model_num)
                vchunk = vectors[a:b]
                norms = l2_norms_sq[a:b]
                disturbing = merging_vector.unsqueeze(0) - vchunk
                inner = torch.matmul(disturbing, vchunk.transpose(1, 2))
                total_loss = total_loss + torch.sum(
                    torch.square(inner) / norms.unsqueeze(-1).unsqueeze(-1)
                )
            total_loss.backward()
            opt.step()
        return merging_vector.data.detach().to(original_dtype).cpu()

    merged_delta = {}
    for name in tvs[0].task_vector_param_dict:
        d0 = tvs[0].task_vector_param_dict[name]
        if d0.dim() == 2 and "lm_head" not in name:
            values = torch.stack([tv.task_vector_param_dict[name] for tv in tvs])
            merged_delta[name] = optimise(name, values)
        else:
            acc = d0.clone()
            for i, tv in enumerate(tvs[1:], 1):
                acc.add_((tv.task_vector_param_dict[name] - acc) / (i + 1))
            merged_delta[name] = acc

    tv = TaskVector(task_vector_param_dict=merged_delta)
    return tv.combine_with_pretrained_model(
        pretrained_model=base_model, scaling_coefficient=scaling_coefficient,
    )


# ----------------------------------------------------------------------------- #
#  Dispatcher                                                                   #
# ----------------------------------------------------------------------------- #

VALID_METHODS = (
    "weight_average", "task_arithmetic", "ties",
    "dare_ta", "dare_ties", "svd", "iso", "wudi",
)


def merge_baseline(method: str,
                   base_model: nn.Module,
                   models_to_merge: List[nn.Module],
                   exclude_param_names_regex: Sequence[str],
                   scaling_coefficient: float = 1.0,
                   ties_param_value_mask_rate: float = 0.8,
                   dare_weight_mask_rate: float = 0.2,
                   wudi_iter_num: int = 300,
                   wudi_lr: float = 1e-5,
                   seed: int = 42,
                   device: str = "cuda") -> dict:
    """Run one paper baseline. Returns a dict of merged params (subset to merge).

    For ``dare_*`` paths, ``torch.manual_seed(seed)`` is set before sampling
    DARE masks (TODO §2.1: single-seed point estimate, no mean/std).
    """
    if method not in VALID_METHODS:
        raise ValueError(f"unknown method {method!r}; must be one of {VALID_METHODS}")

    if method == "weight_average":
        return weight_average_merging(
            base_model, models_to_merge, exclude_param_names_regex,
        )

    if method == "task_arithmetic":
        return task_arithmetic(
            base_model, models_to_merge, exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
        )

    if method == "ties":
        return ties_merging(
            base_model, models_to_merge, exclude_param_names_regex,
            param_value_mask_rate=ties_param_value_mask_rate,
            scaling_coefficient=scaling_coefficient,
        )

    if method in ("dare_ta", "dare_ties"):
        torch.manual_seed(seed)
        masked_models = models_to_merge  # in-place mutation, mirrors original code
        with torch.no_grad():
            for m in masked_models:
                masked = mask_model_weights(
                    finetuned_model=m, pretrained_model=base_model,
                    exclude_param_names_regex=exclude_param_names_regex,
                    weight_format="delta_weight",
                    weight_mask_rate=dare_weight_mask_rate,
                    use_weight_rescale=True, mask_strategy="random",
                )
                copy_params_to_model(masked, m)
        if method == "dare_ta":
            return task_arithmetic(
                base_model, masked_models, exclude_param_names_regex,
                scaling_coefficient=scaling_coefficient,
            )
        return ties_merging(
            base_model, masked_models, exclude_param_names_regex,
            param_value_mask_rate=ties_param_value_mask_rate,
            scaling_coefficient=scaling_coefficient,
        )

    if method == "svd":
        return svd_merging(
            base_model, models_to_merge, exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient, device=device,
        )

    if method == "iso":
        return iso_merging(
            base_model, models_to_merge, exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient, device=device,
        )

    if method == "wudi":
        return wudi_merging(
            base_model, models_to_merge, exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient,
            iter_num=wudi_iter_num, lr=wudi_lr, device=device,
        )

    raise AssertionError("unreachable")


# ----------------------------------------------------------------------------- #
#  Default hyper-parameters per method (used by runner)                         #
# ----------------------------------------------------------------------------- #

DEFAULT_HPARAMS = {
    "weight_average":  dict(scaling_coefficient=1.0),  # mean-of-experts; scale unused
    "task_arithmetic": dict(scaling_coefficient=0.3),
    "ties":            dict(scaling_coefficient=0.3, ties_param_value_mask_rate=0.8),
    "dare_ta":         dict(scaling_coefficient=0.3, dare_weight_mask_rate=0.2),
    "dare_ties":       dict(scaling_coefficient=0.3, dare_weight_mask_rate=0.2,
                            ties_param_value_mask_rate=0.8),
    "svd":             dict(scaling_coefficient=0.3),
    "iso":             dict(scaling_coefficient=0.3),
    "wudi":            dict(scaling_coefficient=0.3, wudi_iter_num=300, wudi_lr=1e-5),
}
