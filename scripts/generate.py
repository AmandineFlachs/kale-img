#!/usr/bin/env python3
"""Generate an image with MOSS-Image + the kale-img-1 LoRA.

Wrapper around KohakuMossTrainer's sampler (its exports are
ComfyUI-format adapters, loaded by its own pipeline). Falls back to
plain diffusers when --no-lora is passed (base-model checks).
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "trainer" / "KohakuMossTrainer"
WEIGHTS = ROOT / "models" / "MOSS-Image-Base"

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--adapter", help="path to exported *-comfy.safetensors")
ap.add_argument("--out", default="gen.png")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--size", type=int, default=1024)
args = ap.parse_args()

adapter = args.adapter
if not adapter:
    exports = sorted(TRAINER.glob("runs/moss/kale-img-1/export/*-comfy.safetensors"))
    if not exports:
        sys.exit("No exported adapter found — pass --adapter explicitly")
    adapter = str(exports[-1])

name = "kale-img-1-sample"
subprocess.run(
    ["bash", "-c",
     f"source .venv/bin/activate && "
     f"kohakumoss-config new sample --weights '{WEIGHTS}' "
     f"--adapter '{adapter}' --name '{name}' || true"],
    cwd=TRAINER, check=True,
)
subprocess.run(
    ["bash", "-c",
     f"source .venv/bin/activate && "
     f"kogine run scripts/sample/sample.py "
     f"--config configs/user/{name}.py "
     f"--set PROMPT='{args.prompt}' "
     f"--set STEPS={args.steps} "
     f"--set CFG={args.cfg} "
     f"--set OUT='{args.out}'"],
    cwd=TRAINER, check=True,
)
