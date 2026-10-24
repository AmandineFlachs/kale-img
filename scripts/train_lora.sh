#!/usr/bin/env bash
# Train the kale-img-1 style LoRA on MOSS-Image via KohakuMossTrainer.
# Target: single RTX 3090 (24GB, sm86). Run from the repo root on the GPU box.
#
# Dataset: dataset/images/ in kohya layout (image.png + image.txt sidecars).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

TRAINER_DIR="${TRAINER_DIR:-trainer/KohakuMossTrainer}"
WEIGHTS="${WEIGHTS:-models/MOSS-Image-Base}"
DATA_DIR="${DATA_DIR:-dataset/images}"
RUN_NAME="${RUN_NAME:-kale-img-1}"

# 1. Trainer ---------------------------------------------------------------
if [ ! -d "$TRAINER_DIR" ]; then
  git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer "$TRAINER_DIR"
fi
if [ ! -d "$TRAINER_DIR/.venv" ]; then
  (cd "$TRAINER_DIR" && bash scripts/install/install.sh)
fi

# Run a command inside the trainer venv, with the trainer dir as cwd.
KOGINE() { (cd "$TRAINER_DIR" && source .venv/bin/activate && "$@"); }

KOGINE kohakumoss-doctor

# 2. Weights ---------------------------------------------------------------
# bf16 HF layout (~26 GB). For the smaller int8 ComfyUI layout (~15 GB) see
# trainer docs/guides/weights.md and point WEIGHTS at the ComfyUI models dir.
if [ ! -d "$WEIGHTS" ]; then
  (cd "$TRAINER_DIR" && source .venv/bin/activate && \
   hf download OpenMOSS-Team/MOSS-Image-Base --local-dir "$ROOT/$WEIGHTS")
fi

# 3. Config ----------------------------------------------------------------
KOGINE kohakumoss-config new train \
  --weights "$ROOT/$WEIGHTS" \
  --data "$ROOT/$DATA_DIR" \
  --name "$RUN_NAME"

CFG="configs/user/${RUN_NAME}.py"
CFG_DATA="configs/user/${RUN_NAME}_data.py"

# 4. Cache text embeddings + latents (one-time) -----------------------------
KOGINE kogine run scripts/data/inspect_data.py --config "$CFG_DATA"
KOGINE kogine run scripts/data/cache_text.py    --config "$CFG_DATA"
KOGINE kogine run scripts/data/cache_latents.py --config "$CFG_DATA"

# 5. Train ------------------------------------------------------------------
# Override trainer knobs straight through, e.g.:
#   bash scripts/train_lora.sh --set LR=2e-4 --set STEPS=2000
KOGINE kogine run scripts/train/moss.py --config "$CFG" "$@"

echo "Checkpoints + exports: $TRAINER_DIR/runs/moss/$RUN_NAME/"
