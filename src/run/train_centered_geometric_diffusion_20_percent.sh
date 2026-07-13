#!/usr/bin/env bash
set -euo pipefail

# Run from the project root no matter where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Keep training prints and Lightning logs visible while the process is running.
export PYTHONUNBUFFERED=1

# Conservative Mac defaults for the centered geometric training set.
python -u src/train_diffusion_model.py \
  --data-dir data/centered_geometric_bw_256 \
  --output-dir checkpoints/diffusion_centered_geometric \
  --batch-size 8 \
  --num-workers 4 \
  --log-resource-every-n-steps 500 \
  --limit-train-batches 1000 \
  --limit-val-batches 125 \
  "$@"
