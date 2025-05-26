# Usage: python calculate_metrics.py merged_ckpt

from collections import defaultdict
import os
import json
import torch
import argparse
from tqdm import tqdm
import re
from pathlib import Path

from ties_merging import convert_delta_to_ft, state_dict_to_vector, topk_values_mask

def parse_merge_info(file):
    pattern = r"Inputs:\n(.*?)\n\nOutput\((.*?)\):(.*?)$"
    match = re.search(pattern, open(file).read().strip(), re.DOTALL)
    if match:
        inputs = match.group(1).split('\n')
        strategy = match.group(2)
        output_path = match.group(3)
    else:
        inputs, strategy, output_path = None, None, None
    return inputs, strategy, output_path

#----- Metrics Implementation -----#
def L2(xy):
    return torch.sqrt(((xy[0]-xy[1])**2).sum())

def cos_sim(xy):
    return 1 - torch.cosine_similarity(xy[0].unsqueeze(0), xy[1].unsqueeze(0)).item()

def soft_sign_dissimilarity(xy):
    xy_abs_sum = xy.abs().sum(dim=0)
    xy_sum = xy.sum(dim=0)
    
    nonzero_mask = xy_abs_sum != 0
    return 1 - (xy_sum[nonzero_mask] / xy_abs_sum[nonzero_mask]).abs().mean()
    
#----- Metrics Implementation -----#
    
def calculate_metrics(merged_ckpt, reset_thresh=50):
    filepaths, _, _ = parse_merge_info(Path(merged_ckpt) / 'merge_info.txt')
    weights_to_merge = defaultdict(list)
    for filepath in filepaths:
        adapter_path = os.path.join(filepath, 'adapter_model.bin')
        if not os.path.exists(adapter_path):
            adapter_path = os.path.join(filepath, 'mm_projector.bin')
        adapter_weights = torch.load(adapter_path, map_location=torch.device('cpu'))
        
        for key in adapter_weights:
            weights_to_merge[key].append(adapter_weights[key].float())
            
    ft_checks, uniques = convert_delta_to_ft(weights_to_merge)
    
    remove_keys = []
    tv_flat_checks = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    all_checks = tv_flat_checks.clone()
    truncated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    
    ssd, tssd = soft_sign_dissimilarity(all_checks), soft_sign_dissimilarity(truncated_checks)
    l2, cosine_sim = L2(all_checks), cos_sim(all_checks)
    
    with open(Path(merged_ckpt) / 'merge_metrics.txt', 'w') as fout:
        fout.write(f'L2: {l2}\n')
        fout.write(f'Cosine: {cosine_sim}\n')
        fout.write(f'SSD: {ssd}\n')
        fout.write(f'TSSD: {tssd}\n')
        
    print(f'L2: {l2}\n')
    print(f'Cosine: {cosine_sim}\n')
    print(f'SSD: {ssd}\n')
    print(f'TSSD: {tssd}\n')

def main():
    parser = argparse.ArgumentParser(description='Calculate parameter interference metrics')
    parser.add_argument('merged_ckpt', help='Path to the merged checkpoint')
    args = parser.parse_args()

    calculate_metrics(args.merged_ckpt)

if __name__ == '__main__':
    main()
