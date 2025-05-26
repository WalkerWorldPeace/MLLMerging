import torch
import gradio as gr
import os
import json
import shortuuid
import mdtex2html
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

model_path = '/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/checkpoints/multimodal-all-demo'
model_base = '/yeesuanAI05/thumt/cc/checkpoints/vicuna-7b-v1.5'
conv_mode = 'vicuna_v1'
running_dtype = torch.float16
# Model
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, modal_processors, context_len = load_pretrained_model(model_path, model_base, model_name)
print("loading finished")
conversation_lib.default_conversation = conv_templates[conv_mode]
tokenizer.pad_token_id = tokenizer.eos_token_id

collate_fn = DataCollatorForSupervisedDataset(tokenizer, modal_processors, {'vision': {'image_aspect_ratio': 'pad'}})
device=model.device

import matplotlib.pyplot as plt
from IPython.display import Audio, Video, Image, display

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

def parse_text(text, modal_inputs):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    outputs = text
    if 'vision' in modal_inputs.keys():
        image_path = modal_inputs['vision'][0]
        text += f'<br><img src="./file={image_path}" style="display: inline-block;width: 250px;max-height: 400px;"><br>'
        outputs = f'<image>{image_path}</image> ' + outputs
    if 'audio' in modal_inputs.keys():
        audio_path = modal_inputs['audio'][0]
        text += f'<br><audio controls src="./file={audio_path}" type="audio/wav"></audio><br>'
        outputs = f'<audio>{audio_path}</audio> ' + outputs
    if 'video' in modal_inputs.keys():
        video_path = modal_inputs['video'][0]
        text += f'<br><video controls width="500" style="display: inline-block;" src="./file={video_path}"></video><br>'
        outputs = f'<video>{video_path}</video> ' + outputs
    text = text.replace('image: <image>\n','')
    text = text.replace('audio: <audio>\n','')
    text = text.replace('video: <video>\n','')
    text = text.replace('point: <point>\n','')
    text = text.replace("\n", '<br>')
    print(text)
    return text, outputs

def predict(prompt_input, image_path, audio_path, video_path, point_path, chatbot, history, modality_cache, temperature=0, top_p=None, num_beams=1):
    conversations = []
    if history is not None and len(history) != 0:
        for (q, a) in history:
            tmp_conv = [{"from": "human", "value": str(q)}, {"from": "gpt", "value": str(a)}]
            conversations += tmp_conv
    modals = {}
    if point_path is not None:
        prompt_input = "point: <point>\n" + prompt_input
        if not isinstance(point_path, str):
            point_path = point_path.name
        modals['point'] = [point_path]
    if video_path is not None:
        prompt_input = "video: <video>\n" + prompt_input
        modals['video'] = [video_path]
    if audio_path is not None:
        prompt_input = "audio: <audio>\n" + prompt_input
        modals['audio'] = [audio_path]
    if image_path is not None:
        prompt_input = "image: <image>\n" + prompt_input
        modals['vision'] = [image_path]
    # int2str = ['zero', 'one', 'two', 'three', 'four']
    # prompt_input = f"Based on {int2str[len(modals)]} input entities:\n" + prompt_input
    tmp_conv = [{"from": "human", "value": prompt_input}, {"from": "gpt", "value": None}]
    conversations += tmp_conv
    input_data = {
        "id": "test",
        "conversations": conversations,
        "modal_inputs": modals
    }
    print(input_data)
    dataset = MultimodalDataset('data/test/avqa-test_mm_answer.json', tokenizer, None)
    dataset.data = [input_data]
    batched_data = collate_fn([dataset[0]])
    input_ids, modal_inputs = batched_data['input_ids'], batched_data['modal_inputs']
    input_ids = input_ids.to(device=device, non_blocking=True)
    
    for modal in modal_inputs:
        if isinstance(modal_inputs[modal], list):
            for modal_inputs_idx in range(len(modal_inputs[modal])):
                modal_inputs[modal][modal_inputs_idx] = modal_inputs[modal][modal_inputs_idx].to(device=device, non_blocking=True, dtype=running_dtype)
        elif isinstance(modal_inputs[modal], dict):
            for key in modal_inputs[modal]:
                modal_inputs[modal][key] = modal_inputs[modal][key].to(device=device, non_blocking=True, dtype=running_dtype)
        else:
            modal_inputs[modal] = modal_inputs[modal].to(device=device, non_blocking=True, dtype=running_dtype)
    stop_str = conv_templates[conv_mode].sep if conv_templates[conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[conv_mode].sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            modal_inputs=modal_inputs,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
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
    print(outputs)
    
    user_chat, user_outputs = parse_text(prompt_input, modals)
    response_chat = response_outputs = outputs
    chatbot.append((user_chat, response_chat))
    history.append((prompt_input, response_outputs))
    return chatbot, history, modality_cache, None, None, None, None

def re_predict(prompt_input, image_path, audio_path, video_path, point_path, chatbot, history, modality_cache, temperature=0, top_p=None, num_beams=1):
    q, a = history.pop()
    return predict(q, image_path, audio_path, video_path, point_path, chatbot, history, modality_cache, temperature, top_p, num_beams)

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return None, None, None, None, [], [], [], 0, None, 1

with gr.Blocks() as demo:

    gr.HTML("""
        <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; ">Model Composition</h1>
        <h3>This is the demo page of Model Composition, a new paradigm through the model composition of existing MLLMs!</h3>
        <div align="center" style="display: flex;"><a href='https://fzacker.github.io/model-composition/model-composition.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp  &nbsp  &nbsp <img src='https://img.shields.io/badge/Github-Code-blue'></a> &nbsp &nbsp  &nbsp  <a href='https://arxiv.org/abs/2402.12750'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
        """)

    with gr.Row():
        with gr.Column(scale=0.7, min_width=500):
            with gr.Row():
                chatbot = gr.Chatbot(label='Model Composition').style(height=440)

            with gr.Tab("User Input"):
                with gr.Row(scale=3):
                    user_input = gr.Textbox(label="Text", placeholder="Key in something here...", lines=3)
                with gr.Row(scale=3):
                    with gr.Column(scale=1):
                        # image_btn = gr.UploadButton("🖼️ Upload Image", file_types=["image"])
                        image_path = gr.Image(type="filepath", label="Image")  # .style(height=200)  # <PIL.Image.Image image mode=RGB size=512x512 at 0x7F6E06738D90>
                    with gr.Column(scale=1):
                        audio_path = gr.Audio(type='filepath')  #.style(height=200)
                    with gr.Column(scale=1):
                        video_path = gr.Video()  #.style(height=200) # , value=None, interactive=True
                    with gr.Column(scale=1):
                        point_path = gr.File(file_types=['.npy'], label='Point')
        with gr.Column(scale=0.3, min_width=300):
            with gr.Group():
                with gr.Accordion('Text Advanced Options', open=True):
                    temperature = gr.Slider(0, 1, value=0, step=0.01, label="Temperature", interactive=True)
                    top_p = gr.Slider(0, 1, value=None, step=0.01, label="Top P", interactive=True)
                    num_beams = gr.Slider(1, 5, value=1, step=1, label="num_beams", interactive=True)
            with gr.Tab("Operation"):
                with gr.Row(scale=1):
                    submitBtn = gr.Button(value="Submit & Run", variant="primary")
                with gr.Row(scale=1):
                    resubmitBtn = gr.Button("Rerun")
                with gr.Row(scale=1):
                    emptyBtn = gr.Button("Clear History")

    history = gr.State([])
    modality_cache = gr.State([])

    submitBtn.click(
        predict, [
            user_input,
            image_path,
            audio_path,
            video_path,
            point_path,
            chatbot,
            history,
            modality_cache,
            temperature,
            top_p,
            num_beams
        ], [
            chatbot,
            history,
            modality_cache,
            image_path,
            audio_path,
            video_path,
            point_path
        ],
        show_progress=True
    )

    resubmitBtn.click(
        re_predict, [
            user_input,
            image_path,
            audio_path,
            video_path,
            point_path,
            chatbot,
            history,
            modality_cache,
            temperature,
            top_p,
            num_beams
        ], [
            chatbot,
            history,
            modality_cache,
            image_path,
            audio_path,
            video_path,
            point_path
        ],
        show_progress=True
    )

    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[
        image_path,
        audio_path,
        video_path,
        point_path,
        chatbot,
        history,
        modality_cache,
        temperature,
        top_p,
        num_beams
    ], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=24000)
