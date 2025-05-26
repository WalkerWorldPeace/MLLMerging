import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import re
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import math

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge

class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params

def ties_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
    """
    ties merging method (layer-by-layer implementation to save memory)
    :param merged_model: nn.Module, the merged model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
    :param scaling_coefficient: float, scaling coefficient to merge the task vectors
    :return:
    """
    def mask_smallest_magnitude_param_values(param_tensor: torch.Tensor, param_value_mask_rate: float = 0.8):
        """
        Mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate.
        :param param_tensor: Tensor, parameter tensor to mask
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # Convert to float32 to support kthvalue operation
        original_dtype = param_tensor.dtype
        param_tensor = param_tensor.float()
        
        # Calculate the number of parameters to mask
        num_mask_params = int(param_tensor.numel() * param_value_mask_rate)
        
        # Flatten the parameter for kthvalue calculation
        flattened = param_tensor.reshape(-1)
        
        # Calculate the threshold
        kth_value = flattened.abs().kthvalue(k=num_mask_params).values
        
        # Create mask and apply
        mask = param_tensor.abs() >= kth_value
        
        # Apply mask and convert back to original dtype
        return (param_tensor * mask).to(original_dtype)

    def get_param_signs(param_tensors: list):
        """
        get the signs for each parameter, computed over individual models
        :param param_tensors: list of Tensor, parameters from different models
        :return:
        """
        # Calculate the sum of parameter signs
        param_sum = sum(param_tensors)
        param_signs = torch.sign(param_sum)
        
        # Handle the case where sign is zero
        if (param_signs == 0).any():
            # Calculate majority sign
            majority_sign = torch.sign(param_signs.sum())
            param_signs[param_signs == 0] = majority_sign
            
        return param_signs

    def disjoint_merge(param_tensors: list, param_signs: torch.Tensor):
        """
        disjoint merge for a single parameter across models
        :param param_tensors: list of Tensor, parameters from different models
        :param param_signs: Tensor, the signs of parameters
        :return:
        """
        preserved_params = []
        for param in param_tensors:
           # Create mask to preserve elements with the same sign as param_signs
            preserve_mask = ((param_signs > 0) & (param > 0)) | ((param_signs < 0) & (param < 0))
            preserved_params.append(param * preserve_mask)
        
        # Calculate how many models preserve each position
        num_preserved = sum([(p != 0).float() for p in preserved_params])
        
        # Calculate the mean, avoid division by zero
        merged_param = sum(preserved_params) / torch.clamp(num_preserved, min=1.0)
        
        return merged_param

    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    # Get the parameter names to merge
    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()), 
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    # Collect task vectors for all models
    print(f"Creating task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            # Compute difference as task vector
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)
    
    # Process parameters layer by layer
    merged_params = {}
    for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
        with torch.no_grad():
            # Collect task vectors for current parameter from all models
            param_vectors = [task_vector[param_name] for task_vector in models_to_merge_task_vectors]
            
            # Apply mask to each parameter
            masked_param_vectors = [
                mask_smallest_magnitude_param_values(param, param_value_mask_rate) 
                for param in param_vectors
            ]
            
            # Calculate signs
            param_signs = get_param_signs(masked_param_vectors)
            
            # Apply disjoint merge strategy
            merged_delta = disjoint_merge(masked_param_vectors, param_signs)
            
            # Combine merged delta with original model parameter
            merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * merged_delta

    return merged_params

def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    original_dtype = input_tensor.dtype
    input_tensor = input_tensor.float()
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor.to(original_dtype)

def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items()):
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy)

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict

def task_arithmetic(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
        """
        task arithmetic method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # iterate each individual model that needs to be merged
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

def svd_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    SVD merging method that uses Singular Value Decomposition to merge models.
    Args:
        merged_model: nn.Module, the base model to merge into  
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to merge the task vectors
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    # Get the parameter names to merge
    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()), 
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    # Compute task vectors
    print("Computing task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            # Compute difference as task vector
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)
    
    sv_reduction = 1.0 / len(models_to_merge)
    device = torch.device("cuda")
    first_param_name = list(models_to_merge_task_vectors[0].keys())[0]
    original_dtype = models_to_merge_task_vectors[0][first_param_name].dtype
    print("Computing SVD merging...")

    with torch.no_grad():
        merged_task_vector_dict = {}
        # Process each parameter
        for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()
            
            # Check parameter shape
            param_shape = models_to_merge_task_vectors[0][param_name].shape
            
            if len(param_shape) == 2 and param_name == 'lm_head.weight':
                print(f"Processing parameter {param_name}, shape: {param_shape}")
                # Apply SVD merging for 2D tensors
                
                # Create temporary variables to store merged results
                sum_u = None
                sum_s = None
                sum_v = None
                
                # Process each model's task vector
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors):
                    # Move parameter to GPU for computation
                    vec = task_vector_dict[param_name].to(device).float()
                    
                    # Compute SVD
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    
                    # Compute reduced index
                    reduced_index_s = int(s.shape[0] * sv_reduction)
                    
                    # Initialize and prepare storage for the first model
                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    
                    # Store important components for each model
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                
                # Compute final merged parameter
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                
                # Compute merged result and move back to CPU
                merged_param = torch.linalg.multi_dot([
                    u_u, v_u, torch.diag(sum_s), u_v, v_v
                ]).to(original_dtype).cpu()
                
                # Store merged parameter
                merged_task_vector_dict[param_name] = merged_param
                
            else:
                # Use simple averaging for non-2D tensors
                merged_param = models_to_merge_task_vectors[0][param_name].clone()
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                merged_task_vector_dict[param_name] = merged_param

        # Create merged task vector and combine with base model
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model,
            scaling_coefficient=scaling_coefficient
        )
        
    return merged_params

def iso_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    Isotropic SVD merging method.
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]
  
    with torch.no_grad():
        # Sum up the task vectors
        merged_task_vector = models_to_merge_task_vectors[0]
        for index in range(1, len(models_to_merge_task_vectors)):
            merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
    
    for param_name, param_value in merged_task_vector.task_vector_param_dict.items():
        original_dtype = param_value.dtype
        param_value = param_value.cuda().to(torch.float32)
        u, s, v = torch.linalg.svd(param_value, full_matrices=False)
        # Compute the average of all singular values (a scalar)
        avg_singular_value = torch.mean(s)
        # Create a diagonal matrix where all diagonal elements are this average value
        avg_s = torch.diag(torch.full_like(s, avg_singular_value))
        
        merged_param = torch.linalg.multi_dot([
            u, avg_s, v
        ]).to(original_dtype).cpu()
        
        # Store merged parameter
        merged_task_vector.task_vector_param_dict[param_name] = merged_param
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    return merged_params

def wudi_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    Wudi merging method that optimizes a merging vector to minimize interference between task vectors.
    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to apply to the final merged vector
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    def get_redundant_task_vector(param_name, vectors, iter_num=300, num_chunks=2):
        original_dtype = vectors.dtype
        vectors = vectors.float().cuda()
        
        model_num, m, n = vectors.shape
        models_per_chunk = (model_num + num_chunks - 1) // num_chunks  # Ceiling division
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        l2_norms = torch.square(torch.norm(vectors.reshape(model_num, -1), p=2, dim=-1))
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            optimizer.zero_grad()
            total_loss = 0.0
            for chunk_idx in range(num_chunks):
                # Compute the model range for the current chunk
                start_model = chunk_idx * models_per_chunk
                end_model = min((chunk_idx + 1) * models_per_chunk, model_num)
                
                # Get vectors for the current chunk
                vectors_chunk = vectors[start_model:end_model, :, :]
                chunk_norms = l2_norms[start_model:end_model]

                # Compute disturbing vectors (chunk_models, m, n)
                disturbing_vectors = merging_vector.unsqueeze(0) - vectors_chunk
                
                # Compute inner product (chunk_models, m, n) x (chunk_models, n, m) -> (chunk_models, m, m)
                inner_product = torch.matmul(disturbing_vectors, vectors_chunk.transpose(1, 2))
                
                # Compute orthogonality loss
                chunk_loss = torch.sum(torch.square(inner_product) / chunk_norms.unsqueeze(-1).unsqueeze(-1))
                
                # Accumulate loss
                total_loss += chunk_loss
            if i % 10 == 0:
                print(f"Step {i}, loss: {total_loss.item()}")
            # Backpropagation and optimization step
            total_loss.backward()
            optimizer.step()
        return merging_vector.data.detach().to(original_dtype).cpu()
  
    merged_task_vector_dict = {}

    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param
    
    # Create merged task vector and combine with base model
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def wudi_merging2(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    Wudi merging2 method that optimizes a merging vector to minimize interference between task vectors
    
    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to apply to the final merged vector
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
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
            v = v * v_mask  # (n, n)
            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)  # m x n
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)
            del u, s, v, u2, s2, v2, S_matrix, s_mask, v_mask
        low_rank = torch.stack(low_rank_list).to(original_dtype)
        taskvector = torch.stack(taskvector_list).to(original_dtype)

        merging_vector = torch.nn.Parameter(average_vector.to(original_dtype))
        optimizer = torch.optim.SGD([merging_vector], lr=1e-4, momentum=0.9)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)).to(original_dtype)
        del vectors, low_rank_list, taskvector_list
        torch.cuda.empty_cache()
      
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return merging_vector.data.detach().cpu()
       
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    # Create merged task vector and combine with base model
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def merge_models(merge_method="wudi2", scaling_coefficient = 1.0):
    print("Start merging models...")
    base_model = models['a']#.cuda()
    base_state_dict = base_model.state_dict()
    models_to_merge = []
    for k in ['b', 'c', 'd', 'e', 'f']:
        model = models[k]#.cuda()
        models_to_merge.append(model)
    
    exclude_param_names_regex = [
        'visual..*',
        '.*embed_tokens.*',
        '.*lm_head.*',
        '.*norm.*',
        '.*bias.*'
    ]
    
    if merge_method == "task_arithmetic":
        print("Running task_arithmetic...")
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "ties":
        print("Running ties_merging...")
        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "dare ta":
        print("Running Dare task_arithmetic...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                # for each individual model, mask its weight
                masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=base_model,
                                                        exclude_param_names_regex=exclude_param_names_regex, weight_format="delta_weight",
                                                        weight_mask_rate=weight_mask_rate, use_weight_rescale=True, mask_strategy="random")
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=new_models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "dare ties":
        print("Running Dare ties_merging...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                # for each individual model, mask its weight
                masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=base_model,
                                                        exclude_param_names_regex=exclude_param_names_regex, weight_format="delta_weight",
                                                        weight_mask_rate=weight_mask_rate, use_weight_rescale=True, mask_strategy="random")
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        
        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=new_model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "svd":
        print("Running tsv_merging...")
        merged_params = svd_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "iso":
        print("Running iso_merging...")
        merged_params = iso_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi":
        print("Running wudi_merging...")
        merged_params = wudi_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi2":
        print("Running wudi v2...")
        merged_params = wudi_merging2(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")

    for key in merged_params:
        if key in base_state_dict:
            base_state_dict[key] = merged_params[key]
    base_model.load_state_dict(base_state_dict)
    base_model = base_model.cuda()

    output_path = 'merged_model_name'
    print(f"Saving model to {output_path}")
    base_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    for model in models_to_merge:
        del model
    torch.cuda.empty_cache()
    return base_model

#####################################################################
path_a = 'Qwen/Qwen2-VL-7B'
path_b = 'yongxianwei/Qwen2-VL-7B-OCR'
path_c = 'yongxianwei/Qwen2-VL-7B-VQA'
path_d = 'yongxianwei/Qwen2-VL-7B-Geometry'
path_e = 'yongxianwei/Qwen2-VL-7B-Chart'
path_f = 'yongxianwei/Qwen2-VL-7B-Grounding'

processor = AutoProcessor.from_pretrained(path_a)
models = {
    'a': Qwen2VLForConditionalGeneration.from_pretrained(path_a, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'b': Qwen2VLForConditionalGeneration.from_pretrained(path_b, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'c': Qwen2VLForConditionalGeneration.from_pretrained(path_c, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'd': Qwen2VLForConditionalGeneration.from_pretrained(path_d, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'e': Qwen2VLForConditionalGeneration.from_pretrained(path_e, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'f': Qwen2VLForConditionalGeneration.from_pretrained(path_f, torch_dtype=torch.float16, trust_remote_code=True).eval(),
}
model = merge_models()
#####################################################################
# Set example image paths
image_path1 = 'examples/image1.jpg'
image_path2 = 'examples/image2.jpg'
import os
abs_image_path1 = os.path.abspath(image_path1)
abs_image_path2 = os.path.abspath(image_path2)

print("\n=== Single Image Example ===")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{abs_image_path1}"},
            {"type": "text", "text": "Please describe the image shortly."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generation_config = dict(max_new_tokens=2048, do_sample=True)
generated_ids = model.generate(**inputs, **generation_config)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Model answer: {output_text[0]}")

print("\n=== Multi-turn Dialogue Example ===")
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{abs_image_path1}"},
            {"type": "text", "text": "Please describe the image in detail."},
        ],
    }
]

text = processor.apply_chat_template(
    messages1, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages1)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, **generation_config)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
first_response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f"Model answer {first_response}")

messages2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{abs_image_path1}"},
            {"type": "text", "text": "Please describe the image in detail."},
        ],
    },
    {
        "role": "assistant",
        "content": first_response,
    },
    {
        "role": "user",
        "content": "Please write a poem according to the image.",
    }
]

text = processor.apply_chat_template(
    messages2, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages2)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, **generation_config)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
poem_response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f"Model answer (poem): {poem_response}")

print("\n=== Multiple Images Example ===")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{abs_image_path1}"},
            {"type": "image", "image": f"file://{abs_image_path2}"},
            {"type": "text", "text": "What are the similarities and differences between these two images."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, **generation_config)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Model answer: {output_text[0]}")

print("\n=== Pure Text Dialogue Example ===")
messages = [
    {
        "role": "user",
        "content": "Hello, who are you? Can you tell me a story?",
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, **generation_config)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Model answer: {output_text[0]}")