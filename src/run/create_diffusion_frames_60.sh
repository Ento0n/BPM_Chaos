#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
EASING="${EASING:-logarithmic}"

python src/generate_diffusion_interpolation_frames.py \
    --checkpoint-dir checkpoints/diffusion \
    --checkpoint checkpoints/diffusion/1/diffusion-epoch=02-val_loss=0.0105.ckpt \
    --run-parent-dir generated \
    --num-beats 80 \
    --image-size 256 \
    --fps 30 \
    --bpm 160 \
    --num-inference-steps 100 \
    --scheduler ddim \
    --interpolation slerp \
    --easing "${EASING}" \
    --accelerator mps \
    --require-device mps \
    --seed 42
