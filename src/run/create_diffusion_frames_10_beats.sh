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
  --checkpoint /Users/antonspannagl/PythonProjects/BPM_Chaos/checkpoints/mixed_diffusion/centered_geometric_wood_diagonals_alpha_0.50.ckpt \
  --run-parent-dir generated \
  --num-beats 10 \
  --image-size 256 \
  --fps 30 \
  --bpm 91 \
  --num-inference-steps 100 \
  --scheduler ddim \
  --interpolation slerp \
  --easing "${EASING}" \
  --accelerator mps \
  --require-device mps \
  --seed 42 \
  "$@"
