PYTHONUNBUFFERED=1

conda run -n bpm_chaos python src/generate_diffusion_interpolation_frames.py \
    --checkpoint-dir checkpoints/diffusion \
    --checkpoint checkpoints/diffusion/diffusion-epoch=08-val_loss=0.0118.ckpt \
    --run-parent-dir generated \
    --num-beats 60 \
    --image-size 256 \
    --fps 30 \
    --bpm 120 \
    --num-inference-steps 100 \
    --scheduler ddim \
    --interpolation slerp \
    --easing cosine \
    --accelerator mps \
    --require-device mps \
    --seed 42
