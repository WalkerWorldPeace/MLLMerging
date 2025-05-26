import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# Set to avoid verbose printing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from modelcompose.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from modelcompose import conversation as conversation_lib
from modelcompose.conversation import conv_templates, SeparatorStyle
from modelcompose.model.builder import load_pretrained_model
from modelcompose.utils import disable_torch_init
from modelcompose.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from modelcompose.data.multimodal_dataset import MultimodalDataset, DataCollatorForSupervisedDataset
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class ChunkedMultimodalDataset(MultimodalDataset):
    
    def __init__(
        self,
        data_path,
        tokenizer,
        data_args,
        num_chunks=1,
        chunk_idx=0
    ):
        super().__init__(data_path, tokenizer, data_args)
        self.data = get_chunk(self.data, num_chunks, chunk_idx)


def create_data_loader(data_path, tokenizer, modal_processors, num_chunks=1, chunk_idx=0, batch_size=1, num_workers=4):
    dataset = ChunkedMultimodalDataset(data_path, tokenizer, None, num_chunks, chunk_idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=DataCollatorForSupervisedDataset(tokenizer, modal_processors, {'vision': {'image_aspect_ratio': 'pad'}}))
    return data_loader


def eval_model(args):
    # CONFIG
    # running_dtype = torch.bfloat16
    running_dtype = torch.float16
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, modal_processors, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    conversation_lib.default_conversation = conv_templates[args.conv_mode]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    data_loader = create_data_loader(args.question_file, tokenizer, modal_processors, args.num_chunks, args.chunk_idx)
    
    questions = get_chunk(json.load(open(args.question_file)), args.num_chunks, args.chunk_idx)
    
    for idx, batched_data in enumerate(tqdm(data_loader)):
        input_ids, modal_inputs = batched_data['input_ids'], batched_data['modal_inputs']
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        for modal in modal_inputs:
            if isinstance(modal_inputs[modal], list):
                for modal_inputs_idx in range(len(modal_inputs[modal])):
                    modal_inputs[modal][modal_inputs_idx] = modal_inputs[modal][modal_inputs_idx].to(device='cuda', non_blocking=True, dtype=running_dtype)
            elif isinstance(modal_inputs[modal], dict):
                for key in modal_inputs[modal]:
                    modal_inputs[modal][key] = modal_inputs[modal][key].to(device='cuda', non_blocking=True, dtype=running_dtype)
            else:
                modal_inputs[modal] = modal_inputs[modal].to(device='cuda', non_blocking=True, dtype=running_dtype)
        
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                modal_inputs=modal_inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": questions[idx]['id'],
                                   "prompt": questions[idx]['conversations'][0]['value'],
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_add_image_token", action="store_true")
    args = parser.parse_args()

    if args.model_base == '' or args.model_base == 'None':
        args.model_base = None

    eval_model(args)
