import os
import json
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict

from peft.utils import transpose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_base')
    parser.add_argument('--image_checkpoint')
    parser.add_argument('--audio_checkpoint')
    parser.add_argument('--output_path')
    parser.add_argument('--strategy', default='avg')
    
    return parser.parse_args()

def load_pretrained_config(path):
    config = json.load(
        open(os.path.join(path, 'config.json'))
    )
    return config

def load_pretrained_weights(path):
    weights_1 = torch.load(os.path.join(path, 'pytorch_model-00001-of-00002.bin'))
    weights_2 = torch.load(os.path.join(path, 'pytorch_model-00002-of-00002.bin'))
    
    weights_1.update(weights_2)
    return weights_1, weights_1.keys(), weights_2.keys()

def get_delta_weight(lora_A_weight, lora_B_weight, scale, fan_in_fan_out=False):
    return (
        transpose(
            lora_B_weight @ lora_A_weight,
            fan_in_fan_out,
        )
        * scale
    )

def lora_key_to_base_key(lora_key):
    return lora_key.replace('base_model.model.', '').replace('.lora_A.weight', '').replace('.lora_B.weight', '') + '.weight'

def base_key_to_lora_key(base_key):
    lora_A_key = 'base_model.model.' + base_key.replace('.weight', '.lora_A.weight')
    lora_B_key = 'base_model.model.' + base_key.replace('.weight', '.lora_B.weight')
    return lora_A_key, lora_B_key
    
def load_delta_weights(path):
    config = json.load(
        open(os.path.join(path, 'config.json'))
    )
    
    adapter_config = json.load(
        open(os.path.join(path, 'adapter_config.json'))
    )
    scale = adapter_config['lora_alpha'] / adapter_config['r']
    
    lora_weights = torch.load(os.path.join(path, 'adapter_model.bin'))
    
    all_base_keys = sorted(list(set([lora_key_to_base_key(lora_key) for lora_key in list(lora_weights.keys())])))
    
    all_delta_weights = {}
    for base_key in all_base_keys:
        lora_A_key, lora_B_key = base_key_to_lora_key(base_key)
        lora_A_weight, lora_B_weight = lora_weights[lora_A_key], lora_weights[lora_B_key]
        all_delta_weights[base_key] = get_delta_weight(lora_A_weight, lora_B_weight, scale)
    
    additional_weights = torch.load(os.path.join(path, 'non_lora_trainables.bin'))
    additional_weights = {
        lora_key_to_base_key(k): v for k, v in additional_weights.items()
    }
    
    return all_delta_weights, additional_weights, config

def maybe_convert_to_multimodal(additional_weights, modal='vision'):
    new_additional_weights = {}
    for k, v in additional_weights.items():
        if not k.starts_with('model.modal_projectors'):
            assert k.starts_with('model.mm_projector')
            k = k[18:] + f'model.modal_projectors.{modal}'
        new_additional_weights[k] = v
    return new_additional_weights

def main(args):
    delta_weights = defaultdict(list)
    additional_weights = dict()
    
    base_config = load_pretrained_config(args.model_base)
    
    if args.image_checkpoint:
        image_delta_weights, image_additional_weights, image_config = load_delta_weights(args.image_checkpoint)
        for k, v in image_delta_weights.items():
            delta_weights[k].append(v)
        
        image_additional_weights = maybe_convert_to_multimodal(additional_weights, modal='vision')
        additional_weights.update(image_additional_weights)
        
        # base_config.mm_vision_tower = image_config.mm_vision_tower
        base_config = image_config
        
    if args.audio_checkpoint:
        audio_delta_weights, audio_additional_weights, audio_config = load_delta_weights(args.audio_checkpoint)
        for k, v in audio_delta_weights.items():
            delta_weights[k].append(v)
        additional_weights.update(audio_additional_weights)
        base_config['mm_audio_encoder'] = audio_config['mm_audio_encoder']

    base_weights, weights_1_keys, weights_2_keys = load_pretrained_weights(args.model_base)
    for k in delta_weights:
        if args.strategy == 'avg':
            base_weights[k] += torch.stack(delta_weights[k]).mean(dim=0)
    
    base_weights.update(additional_weights)
    
    os.makedirs(args.output_path, exist_ok=True)
    
    weights_1 = {}
    weights_2 = {}
    for k, v in base_weights.items():
        if k in weights_1_keys:
            weights_1[k] = v
        else:
            weights_2[k] = v
    
    index = {'metadata': {'total_size': 13476839424}, 'weight_map': {}}
    for k in weights_1:
        index['weight_map'][k] = 'pytorch_model-00001-of-00002.bin'
    for k in weights_2:
        index['weight_map'][k] = 'pytorch_model-00002-of-00002.bin'
    
    torch.save(weights_1, os.path.join(args.output_path, 'pytorch_model-00001-of-00002.bin'))
    torch.save(weights_2, os.path.join(args.output_path, 'pytorch_model-00002-of-00002.bin'))
    json.dump(index, open(os.path.join(args.output_path, 'pytorch_model.bin.index.json'), 'w'), indent=4, sort_keys=True)
    
    json.dump(base_config, open(os.path.join(args.output_path, 'config.json'), 'w'), indent=4, sort_keys=True)
    json.dump({'image': args.image_checkpoint, 'audio': args.audio_checkpoint}, open(os.path.join(args.output_path, 'merge_info.json'), 'w'), indent=4, sort_keys=True)
    
    for file in os.listdir(args.model_base):
        if file in ['special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json']:
            os.symlink(os.path.join(args.model_base, file), os.path.join(args.output_path, file))
    
    
if __name__ == '__main__':
    main(parse_args())