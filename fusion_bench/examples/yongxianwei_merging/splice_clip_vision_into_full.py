"""
Splice a merged CLIPVisionModel into the base CLIPModel and save as a full
CLIPModel directory (with tokenizer + processor), so it can be used as
`_pretrained_` in `CLIPVisionModelPool`/`CLIPClassificationMixin` workflows.

Usage:
    python examples/yongxianwei_merging/splice_clip_vision_into_full.py \
        --base openai/clip-vit-base-patch32 \
        --vision outputs/yongxianwei_merging/merged_clip/aswudi_b32 \
        --out outputs/yongxianwei_merging/merged_clip/aswudi_b32_full
"""

import argparse
import os

from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"Loading base CLIPModel: {args.base}")
    full = CLIPModel.from_pretrained(args.base)
    print(f"Loading merged CLIPVisionModel: {args.vision}")
    merged = CLIPVisionModel.from_pretrained(args.vision)

    # The HF CLIPVisionModel wraps `vision_model`. The full CLIPModel exposes
    # `vision_model` directly. They should have identical state_dicts apart
    # from the prefix.
    src_sd = merged.vision_model.state_dict()
    missing, unexpected = full.vision_model.load_state_dict(src_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"strict load failed: missing={missing}, unexpected={unexpected}"
        )

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving full CLIPModel to {args.out}")
    full.save_pretrained(args.out)
    print("Saving processor (tokenizer + image processor)")
    proc = CLIPProcessor.from_pretrained(args.base)
    proc.save_pretrained(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
