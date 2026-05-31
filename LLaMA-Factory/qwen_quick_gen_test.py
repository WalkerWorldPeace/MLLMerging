"""Minimal Qwen2-VL load + generate test — uses VLMEvalKit's generate path."""
import sys
from pathlib import Path
# VLMEvalKit lives at ../VLMEvalKit relative to this script; override with VLMEVALKIT_PATH.
import os
_default_vlme = Path(__file__).resolve().parent.parent / "VLMEvalKit"
sys.path.insert(0, os.environ.get("VLMEVALKIT_PATH", str(_default_vlme)))

import argparse
import time
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
args = ap.parse_args()

print(f"Loading {args.ckpt} ...", flush=True)
t0 = time.time()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.ckpt, torch_dtype="auto", device_map="cpu", attn_implementation="eager"
)
model.cuda().eval()
processor = Qwen2VLProcessor.from_pretrained(args.ckpt)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

img = Image.new("RGB", (224, 224), color=(255, 0, 0))
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What color is this image? One word."},
    ],
}]
text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)

# Use process_vision_info like VLMEvalKit does
from qwen_vl_utils import process_vision_info
images, videos = process_vision_info([messages])
inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
inputs = inputs.to("cuda")

print(f"input_ids shape: {inputs.input_ids.shape}", flush=True)
print(f"input_ids[0, :20]: {inputs.input_ids[0, :20].tolist()}", flush=True)

print("Generating ...", flush=True)
try:
    out = model.generate(**inputs, max_new_tokens=12, top_p=0.001, top_k=1, temperature=0.01)
    gen = out[0, inputs.input_ids.shape[1]:]
    text_out = processor.tokenizer.decode(gen, skip_special_tokens=True)
    print(f"GENERATED: {text_out!r}", flush=True)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
