#   Convert checkpoint from llava-format to multimodal-format
#   - For LLaVA testing, use llava.eval.model_vqa_loader with jsonlines {'question_id', 'image', 'text'}
#   - For multimodal testing, use llava.eval.model_multimodal_qa_loader with json List[{'id', 'conversations', 'modal_inputs}]
#
#   Usage:
#   python convert_test_data.py --input_path LLAVA_VAL_FILE --input_image_path IMAGE_PATH --output_path MULTIMODAL_VAL_FILE

import os
import shutil
import json
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict

from peft.utils import transpose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('llava_checkpoint')
    parser.add_argument('--output_path')
    
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

def lora_key_to_base_key(lora_key):
    return lora_key.replace('base_model.model.', '').replace('.lora_A.weight', '').replace('.lora_B.weight', '') + '.weight'

def base_key_to_lora_key(base_key):
    lora_A_key = 'base_model.model.' + base_key.replace('.weight', '.lora_A.weight')
    lora_B_key = 'base_model.model.' + base_key.replace('.weight', '.lora_B.weight')
    return lora_A_key, lora_B_key

def llava_key_to_multimodal_key(llava_key):
    if 'lora_A.default' in llava_key or 'lora_B.default' in llava_key:
        return llava_key.replace('default', 'vision')
    if 'mm_projector' in llava_key:
        return llava_key.replace('mm_projector', 'modal_projectors.vision')
    if 'prefix_tokens' in llava_key:
        return llava_key.replace('prefix_tokens', 'prefix_tokens.vision')
    if 'suffix_tokens' in llava_key:
        return llava_key.replace('suffix_tokens', 'suffix_tokens.vision')
    return None

def maybe_convert_to_multimodal(additional_weights, modal='vision'):
    new_additional_weights = {}
    for k, v in additional_weights.items():
        if not k.starts_with('model.modal_projectors'):
            assert k.starts_with('model.mm_projector')
            k = k[18:] + f'model.modal_projectors.{modal}'
        new_additional_weights[k] = v
    return new_additional_weights

def main(args):
    adapter_weights = dict()
    non_lora_trainables = dict()
    
    weights, _, _ = load_pretrained_weights(args.llava_checkpoint)
    for llava_key in tqdm(weights):
        converted_key = llava_key_to_multimodal_key(llava_key)
        if not converted_key:
            continue
        if 'lora' in converted_key:
            adapter_weights[converted_key] = weights[llava_key]
        else:
            non_lora_trainables[converted_key] = weights[llava_key]
    os.makedirs(args.output_path, exist_ok=True)
    torch.save(adapter_weights, os.path.join(args.output_path, 'adapter_model.bin'))
    torch.save(non_lora_trainables, os.path.join(args.output_path, 'non_lora_trainables.bin'))
    
    for file in os.listdir(args.llava_checkpoint):
        if file in ['special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json', 'config.json']:
            shutil.copy(os.path.join(args.llava_checkpoint, file), os.path.join(args.output_path, file))
    
if __name__ == '__main__':
    main(parse_args())