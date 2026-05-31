"""CLI runner for OptMerge / wudi v2 on Qwen2-VL-7B multi-expert merging.

Mirrors run_merge.py but calls wudi_merging2 (the OptMerge paper's method)
instead of SWUDI / ASWUDI. Implementation matches
``MLLMerging/LLaMA-Factory/model_merging.py:542`` (wudi_merging2) byte-for-byte:
SGD lr=1e-4 momentum=0.9, average_vector init, raw SVD (no demean).

Default expert pool follows the OptMerge paper:
    base       Qwen/Qwen2-VL-7B
    experts    yongxianwei/Qwen2-VL-7B-{OCR,VQA,Geometry,Chart,Grounding}

The exclude_param_names_regex is identical to
``MLLMerging/LLaMA-Factory/model_merging.py``: visual.*, embed_tokens,
lm_head, norm, bias -- only LLM body 2-D matrices are merged.

Usage:
    python run_merge_wudi2.py --output_path merged/optmerge_s10  # scale=1.0 default
    python run_merge_wudi2.py --scaling_coefficient 0.1 --output_path merged/optmerge_s01
"""

import argparse
import os
import re

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


DEFAULT_BASE = "Qwen/Qwen2-VL-7B"
DEFAULT_EXPERTS = [
    "yongxianwei/Qwen2-VL-7B-OCR",
    "yongxianwei/Qwen2-VL-7B-VQA",
    "yongxianwei/Qwen2-VL-7B-Geometry",
    "yongxianwei/Qwen2-VL-7B-Chart",
    "yongxianwei/Qwen2-VL-7B-Grounding",
]
DEFAULT_EXCLUDE_REGEX = [
    "visual..*",
    ".*embed_tokens.*",
    ".*lm_head.*",
    ".*norm.*",
    ".*bias.*",
]


def get_param_names_to_merge(input_param_names, exclude_param_names_regex):
    out = []
    for name in input_param_names:
        if not any(re.match(pat, name) for pat in exclude_param_names_regex):
            out.append(name)
    return out


class TaskVector:
    def __init__(self, pretrained_model=None, finetuned_model=None,
                 exclude_param_names_regex=None, task_vector_param_dict=None):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pre = dict(pretrained_model.named_parameters())
            fin = dict(finetuned_model.named_parameters())
            names = get_param_names_to_merge(list(pre.keys()), exclude_param_names_regex)
            with torch.no_grad():
                for n in names:
                    self.task_vector_param_dict[n] = fin[n] - pre[n]

    def combine_with_pretrained_model(self, pretrained_model, scaling_coefficient=1.0):
        pre = dict(pretrained_model.named_parameters())
        with torch.no_grad():
            merged = {}
            for n in self.task_vector_param_dict:
                merged[n] = pre[n] + scaling_coefficient * self.task_vector_param_dict[n]
        return merged


def wudi_merging2(merged_model, models_to_merge, exclude_param_names_regex,
                  scaling_coefficient=1.0):
    """Verbatim port of MLLMerging/LLaMA-Factory/model_merging.py:wudi_merging2."""
    assert isinstance(scaling_coefficient, float)
    tv_list = [
        TaskVector(pretrained_model=merged_model, finetuned_model=m,
                   exclude_param_names_regex=exclude_param_names_regex)
        for m in models_to_merge
    ]

    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).cuda()
        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / vectors.shape[0])
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)
            del u, s, v, u2, s2, v2, S_matrix, s_mask, v_mask
        low_rank = torch.stack(low_rank_list).to(original_dtype)
        taskvector = torch.stack(taskvector_list).to(original_dtype)

        merging_vector = torch.nn.Parameter(average_vector.to(original_dtype))
        optimizer = torch.optim.SGD([merging_vector], lr=1e-4, momentum=0.9)
        l2_norms = torch.square(torch.norm(
            vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)).to(original_dtype)
        del vectors, low_rank_list, taskvector_list
        torch.cuda.empty_cache()

        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            disturbing = merging_vector.unsqueeze(0) - taskvector
            inner = torch.matmul(disturbing, low_rank.transpose(1, 2))
            loss = torch.sum(torch.square(inner) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 50 == 0:
                print(f"  Step {i}, loss: {loss.item():.4f}", flush=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return merging_vector.data.detach().cpu()

    merged_dict = {}
    for name in tv_list[0].task_vector_param_dict:
        if (len(tv_list[0].task_vector_param_dict[name].shape) == 2
                and "lm_head" not in name):
            print(f"Processing {name} with shape "
                  f"{tv_list[0].task_vector_param_dict[name].shape}", flush=True)
            values = torch.stack([tv.task_vector_param_dict[name] for tv in tv_list])
            merged_dict[name] = get_redundant_task_vector(name, values, iter_num=300)

    # simple-average for non-2D params (not actually saved -- they're all
    # excluded by regex, but matches reference implementation)
    for name in tv_list[0].task_vector_param_dict:
        if name not in merged_dict:
            print(f"Using simple averaging for {name}", flush=True)
            avg = tv_list[0].task_vector_param_dict[name].clone()
            for i, tv in enumerate(tv_list[1:], 1):
                avg += (tv.task_vector_param_dict[name] - avg) / (i + 1)
            merged_dict[name] = avg

    merged_tv = TaskVector(task_vector_param_dict=merged_dict)
    return merged_tv.combine_with_pretrained_model(
        pretrained_model=merged_model, scaling_coefficient=scaling_coefficient,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", required=True)
    p.add_argument("--scaling_coefficient", type=float, default=1.0,
                   help="OptMerge paper / model_merging.py default for "
                        "Qwen2-VL is 1.0.")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--experts", nargs="+", default=DEFAULT_EXPERTS)
    p.add_argument("--torch_dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.torch_dtype]

    print(f"Loading base from {args.base} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.base)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True,
    ).eval()
    # Reference: kept on CPU (model_merging.py line 647 has #.cuda() commented out).

    finetuned = []
    for path in args.experts:
        print(f"Loading expert from {path} ...", flush=True)
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True,
        ).eval()
        finetuned.append(m)

    merged = wudi_merging2(
        merged_model=base_model,
        models_to_merge=finetuned,
        exclude_param_names_regex=DEFAULT_EXCLUDE_REGEX,
        scaling_coefficient=float(args.scaling_coefficient),
    )

    base_state = base_model.state_dict()
    for k, v in merged.items():
        if k in base_state:
            base_state[k] = v
    base_model.load_state_dict(base_state)

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving merged model to {args.output_path} ...", flush=True)
    base_model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
