"""Minimal Qwen2-VL sanity: load merged checkpoint, ask 'what color' on red image,
print greedy text. No first-token analysis (which fails for some reason)."""
import argparse
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    print(f"Loading {args.ckpt} ...", flush=True)
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(args.ckpt)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What color is this image? Answer in one word."},
        ],
    }]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    gen = out[0, inputs.input_ids.shape[1]:]
    text_out = proc.tokenizer.decode(gen, skip_special_tokens=True)
    print(f"greedy output: {text_out!r}", flush=True)


if __name__ == "__main__":
    main()
