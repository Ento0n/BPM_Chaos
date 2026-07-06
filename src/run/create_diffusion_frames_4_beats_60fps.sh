#!/usr/bin/env bash
set -euo pipefail

# Run from the project root no matter where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
EASING="${EASING:-logarithmic}"

conda run -n bpm_chaos python src/generate_diffusion_interpolation_frames.py \
  --checkpoint-dir checkpoints/diffusion \
  --checkpoint checkpoints/diffusion/diffusion-epoch=08-val_loss=0.0118.ckpt \
  --run-parent-dir generated \
  --num-beats 4 \
  --image-size 256 \
  --fps 60 \
  --bpm 120 \
  --num-inference-steps 100 \
  --scheduler ddim \
  --interpolation slerp \
  --easing "${EASING}" \
  --accelerator mps \
  --require-device mps \
  --seed 42 \
  "$@"
