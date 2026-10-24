#!/usr/bin/env python3
"""Prepare the kale-img-1 style dataset.

For a style LoRA we want every image captioned. This script:
  1. Scans dataset/images/ for images (png/jpg/jpeg/webp).
  2. Creates a .txt caption file next to each image that lacks one,
     or generates captions with Florence-2 if --auto-caption is passed.
  3. Emits dataset/metadata.jsonl (HF imagefolder format) so the
     diffusers Sana trainer picks up per-image captions via
     --dataset_name/--caption_column.

Caption convention: every caption starts with the trigger token `kxv`
so the style is invocable at inference ("kaleimg style illustration of ...").
"""

import argparse
import json
from pathlib import Path

TRIGGER = "kxv"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
IMAGES_DIR = DATASET_DIR / "images"


def auto_caption(image_paths):
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Florence-2-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    for p in image_paths:
        img = Image.open(p).convert("RGB")
        inputs = processor(
            text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt"
        ).to(device, dtype)
        with torch.inference_mode():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        text = processor.batch_decode(ids, skip_special_tokens=False)[0]
        caption = processor.post_process_generation(
            text, task="<MORE_DETAILED_CAPTION>", image_size=img.size
        )["<MORE_DETAILED_CAPTION>"]
        p.with_suffix(".txt").write_text(f"{TRIGGER} style, {caption}")
        print(f"captioned {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-caption", action="store_true",
                    help="Generate captions with Florence-2 instead of stubs")
    args = ap.parse_args()

    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(
            f"No images found in {IMAGES_DIR} — drop your style images there first."
        )

    todo = [p for p in images if not p.with_suffix(".txt").exists()]
    print(f"{len(images)} images, {len(todo)} missing captions")

    if args.auto_caption:
        auto_caption(todo)
    elif todo:
        for p in todo:
            p.with_suffix(".txt").write_text(f"{TRIGGER} style, ")
        print("Created empty caption stubs — fill them in, or rerun with --auto-caption")

    rows = []
    for p in images:
        txt = p.with_suffix(".txt")
        if txt.exists():
            rows.append({"file_name": f"images/{p.name}", "text": txt.read_text().strip()})
    out = DATASET_DIR / "metadata.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
