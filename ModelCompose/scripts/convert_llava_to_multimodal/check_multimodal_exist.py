import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    
    return parser.parse_args()

def main(args):
    data = json.load(open(args.input_path))
    cnt = 0
    for xx in tqdm(data):
        for modal in xx['modal_inputs']:
            for file in xx['modal_inputs'][modal]:
                if not os.path.exists(file):
                    print(xx)
                    cnt += 1
    if cnt == 0:
        print('All Clear!')
        
if __name__ == '__main__':
    main(parse_args())
    