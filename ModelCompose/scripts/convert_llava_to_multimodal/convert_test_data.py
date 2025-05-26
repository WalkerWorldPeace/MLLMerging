#   Convert test data from llava-format to multimodal-format
#   - For LLaVA testing, use llava.eval.model_vqa_loader with jsonlines {'question_id', 'image', 'text'}
#   - For multimodal testing, use llava.eval.model_multimodal_qa_loader with json List[{'id', 'conversations', 'modal_inputs}]
#
#   Usage:
#   python convert_test_data.py --input_path LLAVA_VAL_FILE --input_image_path IMAGE_PATH --output_path MULTIMODAL_VAL_FILE

import os
import json
import argparse
from tqdm import tqdm

from peft.utils import transpose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--input_image_path')
    parser.add_argument('--output_path')
    
    return parser.parse_args()
    
def main(args):
    args.input_image_path = os.path.abspath(args.input_image_path)
    new_data = []
    with open(args.input_path) as fin:
        for line in tqdm(fin):
            example = json.loads(line)
            new_example = {
                'id': example['question_id'],
                'conversations': [
                    {'from': 'human', 'value': '<image>\n' + example['text']},
                    {'from': 'gpt', 'value': None},
                ],
                'modal_inputs': {
                    'vision': [os.path.join(args.input_image_path, example['image'])]
                }
            }
            for k in example:
                if k not in ['question_id', 'text', 'image']:
                    new_example[k] = example[k]
            new_data.append(new_example)
    json.dump(new_data, open(args.output_path, 'w'))
            
    
if __name__ == '__main__':
    main(parse_args())