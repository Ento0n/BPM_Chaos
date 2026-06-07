from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from PIL import Image

from train_diffusion_model import (
    DEFAULT_OUTPUT_DIR,
    DiffusionModule,
    get_default_accelerator,
)


# Define simple defaults for generated images and checkpoint lookup.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = PROJECT_ROOT / "generated" / "diffusion"


def find_checkpoint(checkpoint_dir: Path) -> Path:
    if True:
        return Path("/Users/antonspannagl/PythonProjects/BPM_Chaos/checkpoints/diffusion/diffusion-epoch=08-val_loss=0.0118.ckpt")

    # Prefer the latest checkpoint, then fall back to the newest checkpoint file.
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint

    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def get_device(accelerator: str) -> torch.device:
    # Pick the torch device that matches the requested accelerator.
    if accelerator == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_image(image_tensor: torch.Tensor, output_path: Path) -> None:
    # Convert one generated tensor from [-1, 1] into an 8-bit grayscale PNG.
    image_tensor = image_tensor.detach().cpu().clamp(-1, 1)
    image_tensor = ((image_tensor + 1.0) * 127.5).to(torch.uint8)
    height, width = image_tensor.shape[-2:]
    pixels = bytes(image_tensor.squeeze(0).contiguous().view(-1).tolist())
    image = Image.frombytes("L", (width, height), pixels)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    # Collect the small set of options needed for sampling images.
    parser = argparse.ArgumentParser(description="Generate grayscale DDPM images.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default=get_default_accelerator())
    return parser.parse_args()


def main() -> None:
    # Resolve the checkpoint and output location.
    args = parse_args()
    checkpoint_path = args.checkpoint or find_checkpoint(args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained Lightning checkpoint onto the selected device.
    device = get_device(args.accelerator)
    diffusion_module = DiffusionModule.load_from_checkpoint(
        checkpoint_path,
        learning_rate=1e-4,
        use_gradient_checkpointing=False,
        image_size=args.image_size,
        map_location=device,
    )
    diffusion_module.eval()
    diffusion_module.to(device)

    # Create the same DDPM scheduler used during training and shorten it for sampling.
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    try:
        scheduler.set_timesteps(args.num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(args.num_inference_steps)
    if hasattr(scheduler, "alphas_cumprod"):
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Start from pure random noise and repeatedly denoise it with the trained UNet.
    torch.manual_seed(args.seed)
    images = torch.randn(
        (args.num_images, 1, args.image_size, args.image_size),
        device=device,
    )

    # Run the reverse diffusion process without gradient tracking.
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            timestep = timestep.to(device)
            timesteps = timestep.repeat(args.num_images)
            predicted_noise = diffusion_module.model(images, timesteps).sample
            images = scheduler.step(predicted_noise, timestep, images).prev_sample

    # Save every generated sample as a black-and-white PNG.
    for index, image_tensor in enumerate(images):
        output_path = args.output_dir / f"sample_{index:03d}.png"
        save_image(image_tensor, output_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved {args.num_images} images to {args.output_dir}")


if __name__ == "__main__":
    main()
