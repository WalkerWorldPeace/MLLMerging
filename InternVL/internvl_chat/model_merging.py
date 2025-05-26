import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import re
import time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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
    def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
        """
        convert parameter dictionary in task vector to a single vector
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        # Tensor, shape (num_total_params, )
        return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

    def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
        """
        convert a single vector to parameter dictionary in task vector
        :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

        return sorted_task_vector_param_dict

    def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
        """
        mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params)
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # Convert to float32 to support kthvalue operation
        flattened_models_to_merge_param = flattened_models_to_merge_param.float()
        
        num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)
        
        # Calculate the threshold
        kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
        
        # Create mask and apply
        mask = flattened_models_to_merge_param.abs() >= kth_values
        
        # Apply mask and convert back to original dtype
        return (flattened_models_to_merge_param * mask).to(flattened_models_to_merge_param.dtype)

    def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
        """
        get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
        # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
        majority_sign = torch.sign(param_signs.sum(dim=0))
        param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
        """
        disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
        # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
        param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

        # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
        num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
        # Tensor, shape (num_total_params, ), the averaged flattened parameters
        merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

        return merged_flattened_param

    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

    flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector=task_vector) for task_vector in models_to_merge_task_vectors]
    # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
    flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

    with torch.no_grad():
        # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
        flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate)

        # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
        param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

        # Tensor, shape (num_total_params, ), disjoint merge
        merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)

        # merged parameter dictionary
        merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(single_vector=merged_flattened_param, task_vector=models_to_merge_task_vectors[0])
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

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
    ISO merging method, uses SVD and equalizes singular values to reduce interference between task vectors

    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, models to be merged
        exclude_param_names_regex: list, regex patterns for parameter names to exclude
        scaling_coefficient: float, scaling coefficient for merging task vectors
    Returns:
        dict: merged parameter dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        # Get parameter shape from the first task vector
        param_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
        
        if len(param_shape) == 2:
            # For 2D parameters, perform SVD merging
            with torch.no_grad():
                merged_param_value = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
                for index in range(1, len(models_to_merge_task_vectors)):
                    merged_param_value = merged_param_value + models_to_merge_task_vectors[index].task_vector_param_dict[param_name]
            
            # SVD and equalize singular values
            original_dtype = merged_param_value.dtype
            merged_param_value = merged_param_value.cuda().to(torch.float32)
            u, s, v = torch.linalg.svd(merged_param_value, full_matrices=False)
            avg_singular_value = torch.mean(s)
            avg_s = torch.diag(torch.full_like(s, avg_singular_value))
            
            merged_param = torch.linalg.multi_dot([
                u, avg_s, v
            ]).to(original_dtype)
            
            # Store merged parameter
            merged_task_vector_dict[param_name] = merged_param
        else:
            # For non-2D parameters, compute the average of all task vectors
            print(param_name)
            with torch.no_grad():
                merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
                for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector.task_vector_param_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                
                merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    return merged_params

def wudi_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    Wudi merging method that optimizes a merging vector to minimize interference between task vectors
    
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
        """
        Optimize a merging vector to minimize interference between task vectors
        
        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).cuda()
       
        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        
        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        
        # Compute L2 norms for normalization
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
       
        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - vectors
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, vectors.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            print(f"Step {i}, loss: {loss.item()}")
            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return merging_vector.data.detach().to(original_dtype)#.cpu()
    
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
        """
        Optimize a merging vector to minimize interference between task vectors
        
        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)

        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
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
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2 + average_vector)
        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)

        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        
        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        
        # Compute L2 norms for normalization
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
       
        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return merging_vector.data.detach().to(original_dtype)
    
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

def merge_models(merge_method="wudi2", scaling_coefficient = 0.1):
    print("Start merging models...")
    base_model = models['a'].cuda()
    base_state_dict = base_model.state_dict()

    models_to_merge = []
    for k in ['b', 'c', 'd', 'e', 'f']:
        model = models[k].cuda()
        models_to_merge.append(model)
    
    exclude_param_names_regex = [
        'vision_model.*',
        '.*lm_head.*',
        '.*norm.*',
        '.*embed_tokens.*',
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
            models_to_merge=new_models_to_merge,
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
    tokenizer.save_pretrained(output_path)
    
    for model in models_to_merge:
        del model
    torch.cuda.empty_cache()
    return base_model

#####################################################################
path_a = 'OpenGVLab/InternVL2_5-1B'
path_b = 'yongxianwei/InternVL2_5-1B_OCR'
path_c = 'yongxianwei/InternVL2_5-1B_VQA'
path_d = 'yongxianwei/InternVL2_5-1B_Geometry'
path_e = 'yongxianwei/InternVL2_5-1B_Chart'
path_f = 'yongxianwei/InternVL2_5-1B_Grounding'

tokenizer = AutoTokenizer.from_pretrained(path_a, trust_remote_code=True, use_fast=False)
models = {
    'a': AutoModel.from_pretrained(path_a, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'b': AutoModel.from_pretrained(path_b, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'c': AutoModel.from_pretrained(path_c, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'd': AutoModel.from_pretrained(path_d, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'e': AutoModel.from_pretrained(path_e, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    'f': AutoModel.from_pretrained(path_f, torch_dtype=torch.float16, trust_remote_code=True).eval(),
}
model = merge_models()
#####################################################################
# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=False)

# pure-text conversation
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# single-image single-round conversation
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# single-image multi-round conversation
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, combined images
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                            history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                            history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, separate images
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                            num_patches_list=num_patches_list,
                            history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                            num_patches_list=num_patches_list,
                            history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# batch inference, single image per sample
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                            num_patches_list=num_patches_list,
                            questions=questions,
                            generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')