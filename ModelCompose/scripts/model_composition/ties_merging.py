import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
### Start of Model conversion utils ###
def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True
### End of Model conversion utils ###

### Start of Merge Utils ###
def topk_values_mask(M, K=0.7, return_mask=False):
    if K >= 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements
    
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    ft_checks,
    reset_thresh=None,
    merge_func="dis-mean",
    remove_keys = []
):
    flat_task_checks = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None
    
    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    # convert the flat merged checkpoint to a state dict
    ptm_check = ft_checks[0] # we use ft_checks[0] to pretend ptm_check, since we only need the state_dict keys in them
    merged_state_dict = vector_to_state_dict(merged_tv, ptm_check, remove_keys)
    return merged_state_dict
### End of Merge Utils ###

def task_arithmetic(models_to_merge_task_vectors: list):
    with torch.no_grad():
        # initially a copy of the weights from the first model
        merged_task_vector = copy.deepcopy(models_to_merge_task_vectors[0])
        # Perform element-wise addition of each parameter of each model
        for index in range(1, len(models_to_merge_task_vectors)):
            for key in merged_task_vector.keys():
                if key in models_to_merge_task_vectors[index]:
                    # Element-wise addition of a tensor
                    merged_task_vector[key] = merged_task_vector[key] + models_to_merge_task_vectors[index][key]
                else:
                    print("Wrong keys!")
    return merged_task_vector

def iso_merging(models_to_merge_task_vectors: list):
    with torch.no_grad():
        # Create a result dictionary, initialized as a copy of the first model's weights
        merged_task_vector = copy.deepcopy(models_to_merge_task_vectors[0])
        # Perform element-wise addition for each parameter of each model
        for index in range(1, len(models_to_merge_task_vectors)):
            for key in merged_task_vector.keys():
                if key in models_to_merge_task_vectors[index]:
                    # Perform element-wise addition on tensors
                    merged_task_vector[key] = merged_task_vector[key] + models_to_merge_task_vectors[index][key]
                else:
                    print(f"Key {key} not found in model {index}")
    
    # Process the merged parameters with isomorphic transformation
    merged_result = {}
    for param_name, param_value in merged_task_vector.items():
        # Only process 2D tensors
        if len(param_value.shape) == 2:
            original_dtype = param_value.dtype
            param_value = param_value.cuda().to(torch.float32)
            
            # Perform SVD decomposition
            u, s, v = torch.linalg.svd(param_value, full_matrices=False)
            # Compute the average of all singular values (a scalar)
            avg_singular_value = torch.mean(s)
            # Create a diagonal matrix where all diagonal elements are the average value
            avg_s = torch.diag(torch.full_like(s, avg_singular_value))
            
            # Reconstruct the parameter
            merged_param = torch.linalg.multi_dot([
                u, avg_s, v
            ]).to(original_dtype).cpu()
            
            merged_result[param_name] = merged_param
        else:
            merged_result[param_name] = param_value
    
    return merged_result

def svd_merging(models_to_merge_task_vectors: list):
    sv_reduction = 1.0 / len(models_to_merge_task_vectors)
    device = torch.device("cuda")
    first_param_name = list(models_to_merge_task_vectors[0].keys())[0]
    original_dtype = models_to_merge_task_vectors[0][first_param_name].dtype
    print("Performing SVD merging...")

    with torch.no_grad():
        merged_task_vector_dict = {}
        # Process each parameter
        for param_name in tqdm(models_to_merge_task_vectors[0].keys(), desc="Processing model parameters"):
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()
            
            # Check the shape of the parameter
            param_shape = models_to_merge_task_vectors[0][param_name].shape
            if len(param_shape) == 2 and param_name != 'lm_head.weight':
                print(f"Processing parameter {param_name}, shape: {param_shape}")
                # Apply SVD merging to 2D tensors
                
                # Create temporary variables to store the merged result
                sum_u = None
                sum_s = None
                sum_v = None
                
                # Process task vectors for each model
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors):
                    # Move the parameter to GPU for computation
                    vec = task_vector_dict[param_name].to(device).float()
                    
                    # Perform SVD
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    
                    # Compute the reduced index
                    reduced_index_s = int(s.shape[0] * sv_reduction)
                    
                    # Initialize and prepare storage space for the first model
                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    
                    # Store the important components of each model
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                
                # Compute the final merged parameter
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                
                # Compute the merged result and move it back to CPU
                merged_param = torch.linalg.multi_dot([
                    u_u, v_u, torch.diag(sum_s), u_v, v_v
                ]).to(original_dtype).cpu()
                
                # Store the merged parameter
                merged_task_vector_dict[param_name] = merged_param
                
            else:
                # Use simple averaging for non-2D tensors
                merged_param = models_to_merge_task_vectors[0][param_name].clone()
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                merged_task_vector_dict[param_name] = merged_param
        
    return merged_task_vector_dict

def wudi_merging(models_to_merge_task_vectors: list):
    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        vectors = vectors.cuda()
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            disturbing_vectors = merging_vector.unsqueeze(0) - vectors
            inner_product = torch.matmul(disturbing_vectors, vectors.transpose(1, 2))
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
        return merging_vector.data.detach().cpu()
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].keys():
        if len(models_to_merge_task_vectors[0][param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0][param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    return merged_task_vector_dict

def wudi_merging2(models_to_merge_task_vectors: list):
    def get_redundant_task_vector(param_name, vectors, iter_num=400):
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).cuda()
        average_vector = vectors.mean(dim=0)
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            reduced_index_s = int(s2.shape[0] / 5)
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)
        taskvector = torch.stack(taskvector_list).to(original_dtype)
        vectors = vectors.to(original_dtype)
        merging_vector = torch.nn.Parameter(average_vector.to(original_dtype))
        optimizer = torch.optim.SGD([merging_vector], lr=5e-2, momentum=0.9)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, vectors.transpose(1, 2))
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
        return merging_vector.data.detach().cpu()
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].keys():
        if len(models_to_merge_task_vectors[0][param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0][param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    return merged_task_vector_dict

### Start of Ties Merging ###
def do_merging(ft_checks, merge_method, K=20, lamda=1.0):
    """
        [INPUT]

        ft_checks is a list of dicts, each dict is a state_dict

        K = 20 (bigger K conserve more params)

        lamda = 1

        [OUTPUT]

        merged_state_dict
    """

    remove_keys = []

    # Since we already have the delta weights from LoRA, we don't minus the flat_ptm
    # tv_flat_checks = flat_ft - flat_ptm
    filtered_ft_checks = []
    for check in ft_checks:
        for key in remove_keys:
            if key in check:
                del check[key]
        filtered_ft_checks.append(check)
    # return merged flat task vector
    if merge_method == "ties":
        merged_tv = ties_merging(
        filtered_ft_checks,
        reset_thresh=K
    )
    elif merge_method == "ta":
        merged_tv = task_arithmetic(
            filtered_ft_checks
        )
    elif merge_method == "tsv":
        merged_tv = svd_merging(
            filtered_ft_checks
        )
    elif merge_method == "iso":
        merged_tv = iso_merging(
            filtered_ft_checks
        )
    elif merge_method == "wudi":
        merged_tv = wudi_merging(
            filtered_ft_checks
        )
    elif merge_method == "wudi2":
        merged_tv = wudi_merging2(
            filtered_ft_checks
        )
    else:
        raise ValueError(f"Unknown merge method: {merge_method}")
    merged_state_dict = {}
    for key, value in merged_tv.items():
        merged_state_dict[key] = value * lamda

    return merged_state_dict
### End of Ties Merging ###

def convert_delta_to_ft(delta_weights):
    """
        input: delta_weights in our script.

        output: (ft_checks, unique) .
            ft_checks: a list of state_dicts that can be the input of func `do_merging()`.
            uniques: keys that are unique (that only appear once)
    """

    # first, we check the keys are the same, and get the length.
    N = -1
    for key in delta_weights.keys():
        N = max(N, len(delta_weights[key]))
    assert N > 0

    
    ft_checks = [{} for _ in range(N)]
    uniques = {}
    for key in delta_weights.keys():
        if len(delta_weights[key]) == N:
            for i in range(N):
                ft_checks[i][key] = delta_weights[key][i]
        else:
            assert len(delta_weights[key]) == 1
            uniques[key] = delta_weights[key][0]
    
    return (ft_checks, uniques)


def demo():
    ft_a = {'x': torch.Tensor([1,2,3]), 'y': torch.Tensor([4,5,6])}
    ft_b = {'x': torch.Tensor([-1,2,3]), 'y': torch.Tensor([0,0,0])}
    print(do_merging([ft_a, ft_b], K=0.9)) # bigger K conserve more params

if __name__ == '__main__':
    demo()