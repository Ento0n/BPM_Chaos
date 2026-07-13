#!/usr/bin/env bash
set -euo pipefail

# Run from the project root no matter where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-generated/26_07_03-11_18_06}"
FRAME_SUBDIR="${FRAME_SUBDIR:-frames}"
COLOR_TRANSITION="${COLOR_TRANSITION:-step}"
COLOR_ARGS=(--random-colors-per-beat --color-transition "${COLOR_TRANSITION}")

# Without COLOR_SEED, the renderer generates a seed and includes it in the
# output filename. Set COLOR_SEED=42 when an exactly reproducible rerender is
# preferred over a new output file.
if [[ -n "${COLOR_SEED:-}" ]]; then
  COLOR_ARGS+=(--color-seed "${COLOR_SEED}")
fi

# The renderer assigns one seeded 0/1 color pair to each 30-frame beat at
# 60 FPS and 120 BPM. Set COLOR_TRANSITION=gradient to ease between palettes.
# Extra arguments can override the output configuration.
conda run -n bpm_chaos python src/create_video_from_frames.py \
  --run-dir "${RUN_DIR}" \
  --frame-subdir "${FRAME_SUBDIR}" \
  --gif \
  --fps 60 \
  --bpm 120 \
  "${COLOR_ARGS[@]}" \
  --overwrite \
  "$@"
