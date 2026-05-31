"""Quick sanity check: load a merged Qwen2-VL-7B checkpoint, ask 'what color?'
on a synthetic red image, report greedy output + first-token top1/entropy.

Reproduces the §6.1 sanity table in mllmerging.md for new scales.

Usage:
    python sanity_red.py --ckpt /path/to/merged --ckpt /path/to/another
"""

import argparse
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def red_image(size=224):
    img = Image.new("RGB", (size, size), color=(255, 0, 0))
    return img


def run_one(ckpt: str):
    print(f"\n=== {ckpt} ===", flush=True)
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.float16, attn_implementation="eager",
    ).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    img = red_image()
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

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]  # last position, full vocab
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        top_p, top_i = probs.max(dim=-1)
        top_tok = proc.tokenizer.decode([top_i.item()])

    print(f"  greedy output:   {text_out!r}")
    print(f"  first-token: top1={top_tok!r}({top_p.item():.3f}), entropy={entropy:.3f} nats")

    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    args = ap.parse_args()
    for c in args.ckpt:
        try:
            run_one(c)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
