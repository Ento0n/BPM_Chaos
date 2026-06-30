#!/usr/bin/env bash
set -euo pipefail

# Run from the project root no matter where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Generate 128x128 grayscale samples from the latest saved checkpoint.
python src/generate_diffusion_images.py \
  --checkpoint /Users/antonspannagl/PythonProjects/BPM_Chaos/checkpoints/diffusion/diffusion-epoch=08-val_loss=0.0118.ckpt \
  --output-dir /Users/antonspannagl/PythonProjects/BPM_Chaos/generated/v1 \
  --image-size 256 \
  --num-images 8 \
  "$@"
