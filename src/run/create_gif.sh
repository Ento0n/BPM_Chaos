#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
RUN_DIR="${1:?Usage: bash src/run/create_gif.sh generated/YY_MM_DD-HH_MM_SS [create_video_from_frames args...]}"
shift
FRAME_SUBDIR="${FRAME_SUBDIR:-frames}"

conda run -n bpm_chaos python src/create_video_from_frames.py \
    --run-dir "${RUN_DIR}" \
    --frame-subdir "${FRAME_SUBDIR}" \
    --gif \
    --fps 30 \
    --bpm 120 \
    --overwrite \
    "$@"
