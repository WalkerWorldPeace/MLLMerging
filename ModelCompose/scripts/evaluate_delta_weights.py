import os
import json
import torch
from peft.utils import transpose

from tqdm import tqdm

def get_delta_weight(lora_A_weight, lora_B_weight, scale, fan_in_fan_out=False):
    return (
        transpose(
            lora_B_weight @ lora_A_weight,
            fan_in_fan_out,
        )
        * scale
    )

def load_pretrained_weights(path):
    weights_1 = torch.load(os.path.join(path, 'pytorch_model-00001-of-00002.bin'))
    weights_2 = torch.load(os.path.join(path, 'pytorch_model-00002-of-00002.bin'))
    
    weights_1.update(weights_2)
    return weights_1, weights_1.keys(), weights_2.keys()

def lora_key_to_base_key(lora_key, modal):
    return lora_key.replace(f'.lora_A.{modal}.weight', '').replace(f'.lora_B.{modal}.weight', '') + '.weight'

def base_key_to_lora_key(base_key, modal):
    lora_A_key = base_key.replace('.weight', f'.lora_A.{modal}.weight')
    lora_B_key = base_key.replace('.weight', f'.lora_B.{modal}.weight')
    return lora_A_key, lora_B_key

def load_delta_weights(path, modals=None):
    config = json.load(
        open(os.path.join(path, 'config.json'))
    )
    
    scale = config['lora_alpha'] / config['lora_r']
    
    lora_weights = torch.load(os.path.join(path, 'adapter_model.bin'))
    
    all_delta_weights = {}
    for key in tqdm(lora_weights):
        if 'lora_B' in key:
            modal = key.split('lora_B.')[1].split('.')[0]
            if modals is not None:
                if modal not in modals:
                    continue
            
            if modal not in all_delta_weights:
                all_delta_weights[modal] = {}
            
            base_key = lora_key_to_base_key(key, modal)
            lora_A_key, lora_B_key = base_key_to_lora_key(base_key, modal)
            lora_A_weight, lora_B_weight = lora_weights[lora_A_key], lora_weights[lora_B_key]
            all_delta_weights[modal][base_key] = get_delta_weight(lora_A_weight, lora_B_weight, scale)

    
    return all_delta_weights

## From Ties Merging
import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F

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



pretrain_path = "/yeesuanAI05/thumt/cc/checkpoints/vicuna-7b-v1.5"
lora_path = "/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/checkpoints/multimodal-vicuna-7b-v1.5-vision-locallora-same"
locallora_path = "/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/checkpoints/multimodal-vicuna-7b-v1.5-vision-locallora-v2-5-language-2e-5-zero3"

lora_delta_weights = load_delta_weights(lora_path, ['default'])['default']
locallora_delta_weights = load_delta_weights(locallora_path, ['default'])['default']

pretrain_weights = load_pretrained_weights(pretrain_path)[0]
pretrain_weights_base = {k: v for k, v in pretrain_weights.items() if k in lora_delta_weights}

lora_task_vector = state_dict_to_vector(lora_delta_weights)
locallora_task_vector = state_dict_to_vector(locallora_delta_weights)
pretrain_vector = state_dict_to_vector(pretrain_weights_base)

import ipdb; ipdb.set_trace()