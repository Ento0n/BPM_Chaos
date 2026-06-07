#!/usr/bin/env bash
set -euo pipefail

# Run from the project root no matter where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Conservative Mac defaults; pass extra CLI args after this script to override them.
python src/train_diffusion_model.py \
  --batch-size 8 \
  --num-workers 4 \
  --log-resource-every-n-steps 500 \
  --limit-train-batches 1000 \
  --limit-val-batches 125 \
  "$@"
