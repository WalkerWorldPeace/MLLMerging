import os
import json
import torch

from tqdm import tqdm

from peft.utils import transpose
from ties_merging import convert_delta_to_ft, state_dict_to_vector, topk_values_mask

def soft_sign_dissimilarity(xy):
    xy_abs_sum = xy.abs().sum(dim=0)
    xy_sum = xy.sum(dim=0)
    
    nonzero_mask = xy_abs_sum != 0
    return 1 - (xy_sum[nonzero_mask] / xy_abs_sum[nonzero_mask]).abs().mean()

def soft_sign_dissimilarity_2(x, y):
    xy_abs_sum = x.abs() + y.abs()
    xy_sum = x + y
    
    nonzero_mask = xy_abs_sum != 0
    return 1 - (xy_sum[nonzero_mask] / xy_abs_sum[nonzero_mask]).abs().mean()

def get_delta_weight(lora_A_weight, lora_B_weight, scale, fan_in_fan_out=False):
    return (
        transpose(
            lora_B_weight @ lora_A_weight,
            fan_in_fan_out,
        )
        * scale
    )

def load_llama_weights(path):
    weights_1 = torch.load(os.path.join(path, 'pytorch_model-00001-of-00002.bin'))
    weights_2 = torch.load(os.path.join(path, 'pytorch_model-00002-of-00002.bin'))
    weights_1.update(**weights_2)
    return weights_1

def load_adapter_weights(path, base_weights):
    
    config = json.load(
        open(os.path.join(path, 'config.json'))
    )
    scale = config['lora_alpha'] / config['lora_r']
    
    def lora_key_to_base_key(lora_key):
        return lora_key.replace('.lora_A.weight', '').replace('.lora_B.weight', '') + '.weight'

    def base_key_to_lora_key(base_key):
        lora_A_key = base_key.replace('.weight', '.lora_A.default.weight')
        lora_B_key = base_key.replace('.weight', '.lora_B.default.weight')
        return lora_A_key, lora_B_key
    
    lora_weights = torch.load(os.path.join(path, 'adapter_model.bin'))
    
    new_weights = {}
    for key in tqdm(base_weights):
        lora_A_key, lora_B_key = base_key_to_lora_key(key)
        if lora_A_key in lora_weights and lora_B_key in lora_weights:
            lora_A_weights, lora_B_weights = lora_weights[lora_A_key], lora_weights[lora_B_key]
            new_weights[key] = base_weights[key].clone() + get_delta_weight(lora_A_weights, lora_B_weights, scale)
    return new_weights

# Note: Repolace the paths below with yours
vicuna_weights = load_llama_weights('/path/to/vicuna-7b-v1.5')
llama2chat_weights = load_llama_weights('/path/to/Llama-2-7b-chat-hf')

llava_weights = load_adapter_weights('/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/checkpoints/multimodal-vicuna-7b-v1.5-vision-locallora-same', vicuna_weights)
llava_audio_weights = load_adapter_weights('/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/checkpoints/multimodal-vicuna-7b-v1.5-audio-beats-qformer-locallora-v2-same', vicuna_weights)
llava_video_weights = load_llama_weights('/yeesuanAI05/thumt/cc/checkpoints/Video-LLaVA-7B')

vicuna_weights = {k: vicuna_weights[k] for k in llava_weights}
llama2chat_weights = {k: llama2chat_weights[k] for k in llava_weights}
llava_video_weights =  {k: llava_video_weights[k] for k in llava_weights}

vicuna_weights = state_dict_to_vector(vicuna_weights)
llama2chat_weights = state_dict_to_vector(llama2chat_weights)
llava_weights = state_dict_to_vector(llava_weights)
llava_audio_weights = state_dict_to_vector(llava_audio_weights)
llava_video_weights = state_dict_to_vector(llava_video_weights)

sim_0 = soft_sign_dissimilarity_2(vicuna_weights.float(), llama2chat_weights.float())
sim_1 = soft_sign_dissimilarity_2(vicuna_weights.float(), llava_weights.float())
sim_2 = soft_sign_dissimilarity_2(vicuna_weights.float(), llava_audio_weights.float())
sim_3 = soft_sign_dissimilarity_2(llava_weights.float(), llava_audio_weights.float())

sim_4 = soft_sign_dissimilarity_2(llava_weights.float(), llava_video_weights.float())
sim_5 = soft_sign_dissimilarity_2(llava_audio_weights.float(), llava_video_weights.float())
sim_6 = soft_sign_dissimilarity_2(vicuna_weights.float(), llava_video_weights.float())

import ipdb; ipdb.set_trace()