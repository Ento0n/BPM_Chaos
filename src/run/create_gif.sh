PYTHONUNBUFFERED=1
RUN_DIR="${1:?Usage: bash src/run/create_gif.sh generated/YY_MM_DD-HH_MM_SS}"

conda run -n bpm_chaos python src/create_video_from_frames.py \
    --run-dir "${RUN_DIR}" \
    --output "${RUN_DIR}/videos/diffusion_interpolation_preview.gif" \
    --fps 30 \
    --bpm 120 \
    --overwrite
