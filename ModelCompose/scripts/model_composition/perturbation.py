# Usage: python merge_checkpoints.py chekpoint_path1 checkpoint_path2 -o merged_checkpoint_path

from collections import defaultdict
import os
import json
import torch
import argparse
from tqdm import tqdm

import copy

from ties_merging import do_merging, convert_delta_to_ft
from calculate_metrics import calculate_metrics

MODAL_DICT={'mm_vision_encoder': 'vision',
            'mm_vision_tower': 'vision',
            'mm_audio_encoder': 'audio'}
def get_modal_from_config(config):
    for key in MODAL_DICT:
        if key in config.keys() and isinstance(config[key], str) and len(config[key]) > 0:
            return MODAL_DICT[key]
    assert False, f'No modality is recognized, please check the config.'

def merge_checkpoints(filepaths, output_path, strategy="sum", K=20):
    configs = []
    weights_to_merge = defaultdict(list)
    for filepath in filepaths:
        adapter_path = os.path.join(filepath, 'adapter_model.bin')
        if not os.path.exists(adapter_path):
            adapter_path = os.path.join(filepath, 'mm_projector.bin')
        adapter_weights = torch.load(adapter_path, map_location=torch.device('cpu'))
        modal_config = json.load(open(os.path.join(filepath, 'config.json')))
        configs.append(modal_config)
        
        for key in adapter_weights:
            weights_to_merge[key].append(adapter_weights[key])
    
    if strategy.startswith('convert-'): # convert 'same' training strategy checkpoint to 'modal+language'
        strategy = strategy.replace('convert-', '')
        # change lora_strategy
        for config in configs:
            if 'lora_strategy' in config:
                assert config['lora_strategy'] == 'same'
                config['lora_strategy'] = 'modal+language'
        # get modal types
        modal_types = []
        for config in configs:
            modal_types.append(get_modal_from_config(config))
        # duplicate weights_to_merge
        convert_weights_to_merge = defaultdict(list)
        for key in weights_to_merge:
            if '.default' in key:
                for i in range(len(modal_types)):
                    new_key = key.replace('default', modal_types[i])
                    convert_weights_to_merge[new_key].append(copy.deepcopy(weights_to_merge[key][i]))

        if strategy.startswith('drop-'):
            merge_func = strategy.replace('drop-', 'dis-')
            ft_checks, uniques = convert_delta_to_ft(weights_to_merge)
            merged_weights = do_merging(ft_checks, K=K, merge_func=merge_func)
            merged_weights.update(uniques)
            
            for k in convert_weights_to_merge:
                convert_weights_to_merge[k] = convert_weights_to_merge[k][0]
            merged_weights.update(convert_weights_to_merge)
        else:
            weights_to_merge.update(convert_weights_to_merge)
            
        
    print(strategy, strategy.startswith('ties-'))
    ft_checks = None
    if strategy.startswith('ties-'):
        assert strategy.replace('ties-', '') in ['sum', 'mean', 'max']
        # print(weights_to_merge.keys())
        
        ft_checks, uniques = convert_delta_to_ft(weights_to_merge)
        
        merge_func = strategy.replace('ties-', 'dis-')
        
        merged_weights = do_merging(ft_checks, K=K, merge_func=merge_func)
        merged_weights.update(uniques)
        
        strategy = f"{merge_func}-{K}"
        
        # print(merged_weights.keys())
        assert sorted(weights_to_merge) == sorted(merged_weights), 'the keys should be the same'
        # print(weights_to_merge['model.modal_projectors.vision.0.weight'])
        # print(merged_weights['model.modal_projectors.vision.0.weight'])
        # exit(0)
    else:
        if strategy == 'sum':
            merged_weights = {}
            for key in weights_to_merge:
                merged_weights[key] = sum(weights_to_merge[key])
        elif strategy == 'mean':
            merged_weights = {}
            for key in weights_to_merge:
                merged_weights[key] = sum(weights_to_merge[key]) / len(weights_to_merge[key])
        else:
            print(f"Merge strategy [{strategy}] not implemented, DO NOTHING.")
            # raise NotImplementedError("Merge strategy not implemented")
    
    merged_configs = {}
    for config in configs:
        for key in config:
            if key in merged_configs:
                merged_configs[key] = merged_configs[key] or config[key]
            else:
                merged_configs[key] = config[key]
    
    os.makedirs(output_path, exist_ok=True)
    torch.save(merged_weights, os.path.join(output_path, 'adapter_model.bin'))
    json.dump(merged_configs, open(os.path.join(output_path, 'config.json'), 'w'), indent=4)
    
    with open(os.path.join(output_path, 'merge_info.txt'), 'w') as fout:
        inputs = '\n'.join(filepaths)
        fout.write(f"Inputs:\n{inputs}\n\nOutput({strategy}):{output_path}")
    print(f"Merged checkpoints saved to {output_path}")
    
    # # calculate merge metrics
    # if ft_checks:
    #     calculate_metrics(output_path, reset_thresh=K)

def main():
    parser = argparse.ArgumentParser(description='Merge multiple torch checkpoints')
    parser.add_argument('filepaths', nargs='+', help='List of checkpoint file paths to merge')
    parser.add_argument('-o', '--output', default='merged_checkpoint.pth', help='Output file path')
    parser.add_argument('--strategy', default='sum', help='Merge strategy')
    parser.add_argument('-K', default=20, type=int, help='K for ties-merging')
    args = parser.parse_args()

    merge_checkpoints(args.filepaths, args.output, args.strategy, args.K)

if __name__ == '__main__':
    main()
