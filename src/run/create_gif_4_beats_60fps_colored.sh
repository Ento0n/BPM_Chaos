#!/usr/bin/env bash
set -euo pipefail

# Run from the project root no matter where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-generated/26_07_03-11_18_06}"
FRAME_SUBDIR="${FRAME_SUBDIR:-frames}"

conda run -n bpm_chaos python src/create_video_from_frames.py \
  --run-dir "${RUN_DIR}" \
  --frame-subdir "${FRAME_SUBDIR}" \
  --gif \
  --fps 60 \
  --bpm 120 \
  --color-0 "#abcdd6" \
  --color-1 "#fb05ff" \
  --overwrite \
  "$@"
