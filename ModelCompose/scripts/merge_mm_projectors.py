# Usage: python merge_checkpoints.py checkpoint1.pth checkpoint2.pth -o merged_checkpoint.pth


import torch
import argparse

def merge_checkpoints(filepaths, output_path):
    merged_state = {}
    for filepath in filepaths:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        merged_state.update(checkpoint)

    torch.save(merged_state, output_path)
    print(f"Merged checkpoints saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Merge multiple torch checkpoints')
    parser.add_argument('filepaths', nargs='+', help='List of checkpoint file paths to merge')
    parser.add_argument('-o', '--output', default='merged_checkpoint.pth', help='Output file path')
    args = parser.parse_args()

    merge_checkpoints(args.filepaths, args.output)

if __name__ == '__main__':
    main()
