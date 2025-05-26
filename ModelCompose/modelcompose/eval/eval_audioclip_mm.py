from modelcompose.model import MultimodalLlamaForCausalLM
import transformers
import torch

from dataclasses import dataclass, field
from typing import Dict, Optional

import json

from modelcompose.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from modelcompose.conversation import conv_templates, SeparatorStyle
from modelcompose.model.builder import load_pretrained_model
from modelcompose.train.train_multimodal import smart_tokenizer_and_embedding_resize
from modelcompose.utils import disable_torch_init
from modelcompose.mm_utils import tokenizer_modal_token
from torch.utils.data import Dataset, DataLoader
from modelcompose.data.multimodal_dataset import MultimodalDataset, DataCollatorForSupervisedDataset
from modelcompose import conversation as conversation_lib

from tqdm import tqdm
import os

class Dict2Class(object):
      
        def __init__(self, my_dict):
            
            for key in my_dict:
                setattr(self, key, my_dict[key])

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # multimodal
    mm_vision_encoder: Optional[str] = field(default=None)
    mm_text_encoder: Optional[str] = field(default=None)
    mm_audio_encoder: Optional[str] = field(default=None)

    mm_video_encoder: Optional[str] = field(default=None)
    eva_vit_g_ckpt: Optional[str] = field(default=None)
    qformer_ckpt:  Optional[str] = field(default=None)

    projectors_path: Optional[str] = field(default=None) # saved projectors path
    lora_path: Optional[str] = field(default=None)

    result_path: Optional[str] = field(default=None) # output result file path

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    mm_projector_lr: Optional[float] = None
    
    group_by_modality_length: bool = field(default=False)  

def print_non_fp16_bf16_params(model):
    for name, param in model.named_parameters():
        if param.dtype != torch.float16 and param.dtype != torch.bfloat16:
            print(f"Parameter: {name}, dtype: {param.dtype}")

def eval():
    # Load Model
    global local_rank

    device = 'cuda'

    # parser = transformers.HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    parser = transformers.HfArgumentParser(
        (ModelArguments,))
    model_args,  = parser.parse_args_into_dataclasses()

    

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            # cache_dir=training_args.cache_dir,
            model_max_length=2048,
            # model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            # cache_dir=training_args.cache_dir,
            model_max_length=2048,
            # model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    model = MultimodalLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        # **bnb_model_from_pretrained_args
    )
    model.get_model().initialize_multimodal_modules(
            model_args=model_args,
            # fsdp=training_args.fsdp
        )
    if model_args.lora_path is not None:
        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_args.lora_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    model.to(torch.bfloat16).to(device)

    print_non_fp16_bf16_params(model)
    # return


    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    data_args = {
        "is_multimodal": True,
    }    
    data_args = Dict2Class(data_args)
    dataset = MultimodalDataset(
        data_path="/data/users/air/dyy/data/audiocaps_mm/val_noAns.json", 
        tokenizer=tokenizer,
        data_args=data_args
    )
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    # for data in dataset:
    #     print(data.keys())  # dict_keys(['input_ids', 'labels', 'modal_inputs'])
    
    # for data in data_loader:
        
    #     print(data['modal_inputs'])
    #     return
    #     print(data.keys())  # dict_keys(['input_ids', 'labels', 'modal_inputs'])

    

    # collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, modal_processors=model.get_modal_processors())

    # batch = collator(dataset)

    # print(batch.keys())
    # print(batch['input_ids'].shape)
    # for key in batch.keys():
    #     print(key)
    #     if(type(batch[key]) is torch.Tensor):
    #         print(batch[key].shape)
    #     else:
    #         print(type(batch[key]), len(batch[key]))
    # return


    result_path = os.path.join("/data/users/air/dyy/data/audiocaps_mm/result", "result.txt")
    if model_args.result_path is not None:
        result_path = model_args.result_path

    results = []
    with open(result_path, 'w') as file:
        for data in tqdm(data_loader):
            with torch.inference_mode():
                # print(data["input_ids"].dtype, data["modal_inputs"]['audio'].dtype) # [1, 1, 3, 1, x, y] # int64 float32
                input_ids = data['input_ids'].to(device)
                # print("input_ids:", input_ids)
                modal_inputs = data['modal_inputs']
                modal_inputs['audio'] = [modal_inputs['audio'].squeeze((0,1)).to(torch.bfloat16).to(device)]
                output_ids = model.generate(
                    input_ids,
                    modal_inputs=modal_inputs,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                    use_cache=False
                )
            input_token_len = data['input_ids'].shape[1]

            # print("output_ids", output_ids)
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            # if outputs.endswith(stop_str):
            #     outputs = outputs[:-len(stop_str)]
            # outputs = outputs.strip()
            print("*-*-*-*-*-*-*-*-*-*-*", file=file)
            print(outputs, file=file)
            file.flush()
            results.append(outputs)

    
    print("Done.")
    return


    b = tokenizer.encode("Close your eyes, open your ears and you imagine only based on the sound that: <audio> . Tell me what you've heard.")
    b = torch.Tensor([b,]).long()
    print(b)
    c = model.get_model().embed_tokens(b)
    print(c.shape)
    a = model.generate(input_ids=b,
                       max_length=50,
                        do_sample=True,  # Example additional generation parameters
                        temperature=0.7  # Example additional generation parameters)
    )
    print("generated:", a)
    e = a.flatten().tolist()
    decoded_text = tokenizer.decode(e, skip_special_tokens=True)
    print("Decoded:", decoded_text)
    print("Done.")

if __name__ == "__main__":
    eval()