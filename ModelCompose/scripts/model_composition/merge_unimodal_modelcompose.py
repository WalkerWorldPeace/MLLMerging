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
            'mm_vision2_encoder': 'vision2',
            'mm_vision2_tower': 'vision2',
            'mm_video_encoder': 'video',
            'mm_audio_encoder': 'audio',
            'mm_point_encoder': 'point'}
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
            print(key)
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
            

    ft_checks = None
    if strategy.startswith('merge-'):
        # print(weights_to_merge.keys())
        
        ft_checks, uniques = convert_delta_to_ft(weights_to_merge)
        
        merge_method = strategy.replace("merge-", "")
        
        merged_weights = do_merging(ft_checks, K=K, merge_method=merge_method)
        merged_weights.update(uniques)
        
        # print(merged_weights.keys())
        assert sorted(weights_to_merge) == sorted(merged_weights), 'the keys should be the same'
        # print(weights_to_merge['model.modal_projectors.vision.0.weight'])
        # print(merged_weights['model.modal_projectors.vision.0.weight'])
        # exit(0)
    elif strategy.startswith('online-merge-'):
        merged_weights = dict()
        modal_names = [get_modal_from_config(config) for config in configs]
        for key in weights_to_merge:
            if len(weights_to_merge[key]) == 1:
                print(1,key)
                merged_weights[key] = weights_to_merge[key][0]
            else:
                assert 'default' in key
                for modal_name, weight in zip(modal_names, weights_to_merge[key]):
                    merged_weights[key.replace('default', f'default-{modal_name}')] = weight
                    # print(f'default-{modal_name}')
    else:
        if strategy == 'sum':
            merged_weights = {}
            for key in weights_to_merge:
                merged_weights[key] = sum(weights_to_merge[key])
        elif strategy == 'mean':
            merged_weights = {}
            for key in weights_to_merge:
                merged_weights[key] = sum(weights_to_merge[key]) / len(weights_to_merge[key])
        elif strategy.startswith('weighted-'):
            weights_str = strategy.replace('weighted-', '')
            weight_coefficients = [float(w) for w in weights_str.split('-')]
            
            model_count = max(len(weights_to_merge[key]) for key in weights_to_merge)
            if len(weight_coefficients) != model_count:
                raise ValueError(f"权重系数数量({len(weight_coefficients)})与模型数量({model_count})不匹配")
            
            # total_weight = sum(weight_coefficients)
            # weight_coefficients = [w/total_weight for w in weight_coefficients]
            
            print(f"使用加权合并策略，权重系数: {weight_coefficients}")
            
            merged_weights = {}
            for key in weights_to_merge:
                # 处理不同长度的参数列表
                if len(weights_to_merge[key]) == 1:
                    merged_weights[key] = weights_to_merge[key][0]
                else:
                    weighted_sum = None
                    for i, weight in enumerate(weights_to_merge[key]):
                        if weighted_sum is None:
                            weighted_sum = weight_coefficients[i] * weight
                        else:
                            weighted_sum += weight_coefficients[i] * weight
                    merged_weights[key] = weighted_sum
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
        if strategy.startswith('online-merge-'):
            strategy = strategy.replace('online-merge-', '')
            if strategy.startswith('reset-'):
                merged_configs['reset_scaling_weights'] = strategy.replace('reset-', '')
            else:
                merged_configs['merge_default_weights'] = strategy
    
    for config in configs:
        modal_name = get_modal_from_config(config)
        # lora_r_dict[modal_name] = config.r
        # lora_alpha_dict[modal_name] = config.lora_alpha
        merged_configs[f'{modal_name}_lora_alpha'] = config['lora_alpha']
        merged_configs[f'{modal_name}_lora_r'] = config['lora_r']
    
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
