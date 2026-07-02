PYTHONUNBUFFERED=1

conda run -n bpm_chaos python src/create_video_from_frames.py \
    --frame-dir generated/diffusion_interpolation/26_07_01-16_53_08 \
    --beat-dir generated/diffusion_interpolation_beats/26_07_01-16_53_08 \
    --output generated/videos/diffusion_interpolation_preview.gif \
    --fps 30 \
    --bpm 120 \
    --overwrite