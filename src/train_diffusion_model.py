from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from models.hf_unet_2d import create_model


# Define the simple project defaults used by the command-line interface.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "synthetic_bw_256"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "diffusion"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_default_accelerator() -> str:
    # Prefer Apple Silicon GPU acceleration on a Mac, then fall back gracefully.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def format_gib(num_bytes: int | float) -> str:
    # Convert byte counts into a human-readable GiB string.
    return f"{num_bytes / 1024**3:.2f} GiB"


class GrayscaleImageDataset(Dataset):
    """Loads grayscale images and scales pixel values to [-1, 1]."""

    def __init__(self, image_paths: list[Path], image_size: int) -> None:
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Load each image as one grayscale channel.
        image = Image.open(self.image_paths[index]).convert("L")

        # Keep the UNet input size fixed at 256x256 unless changed by argument.
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert from uint8 pixels in [0, 255] to a tensor in [-1, 1].
        image_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
        image_tensor = image_tensor.view(1, self.image_size, self.image_size)
        return image_tensor / 127.5 - 1.0


class ImageDataModule(pl.LightningDataModule):
    """Splits one image folder into train, validation, and test loaders."""

    def __init__(
        self,
        data_dir: Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset: GrayscaleImageDataset | None = None
        self.val_dataset: GrayscaleImageDataset | None = None
        self.test_dataset: GrayscaleImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        # Find all supported image files in the dataset directory.
        image_paths = sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {self.data_dir}")

        # Shuffle once with a fixed seed so train/val/test splits are repeatable.
        rng = random.Random(self.seed)
        rng.shuffle(image_paths)

        # Split images into 80% train, 10% validation, and 10% test.
        train_end = int(0.8 * len(image_paths))
        val_end = int(0.9 * len(image_paths))
        train_paths = image_paths[:train_end]
        val_paths = image_paths[train_end:val_end]
        test_paths = image_paths[val_end:]

        self.train_dataset = GrayscaleImageDataset(train_paths, self.image_size)
        self.val_dataset = GrayscaleImageDataset(val_paths, self.image_size)
        self.test_dataset = GrayscaleImageDataset(test_paths, self.image_size)

    def print_summary(self, max_epochs: int, accumulate_grad_batches: int) -> None:
        # Print the amount of work implied by the current split and batch settings.
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self.setup("fit")

        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset)
        test_size = len(self.test_dataset)
        train_batches = math.ceil(train_size / self.batch_size)
        optimizer_steps = math.ceil(train_batches / accumulate_grad_batches) * max_epochs

        print(
            "Dataset split: "
            f"train={train_size}, val={val_size}, test={test_size}, "
            f"train_batches_per_epoch={train_batches}, "
            f"estimated_optimizer_steps={optimizer_steps}"
        )

    def train_dataloader(self) -> DataLoader:
        # Shuffle training batches each epoch for better optimization.
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        # Validation uses deterministic ordering and no shuffling.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        # Test uses deterministic ordering and no shuffling.
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class DiffusionModule(pl.LightningModule):
    """Lightning wrapper that trains the UNet to predict added Gaussian noise."""

    def __init__(
        self,
        learning_rate: float,
        use_gradient_checkpointing: bool,
        image_size: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Reuse the Hugging Face UNet defined in src/models/hf_unet_2d.py.
        self.model = create_model(sample_size=image_size)

        # Save memory by recomputing some activations during backward passes.
        if use_gradient_checkpointing and hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

        # The scheduler controls the forward diffusion noising process.
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def _shared_step(self, images: torch.Tensor, stage: str) -> torch.Tensor:
        # Sample random noise and random diffusion timesteps for this batch.
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
        ).long()

        # Add noise to clean images, then ask the UNet to predict that noise.
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        predicted_noise = self.model(noisy_images, timesteps).sample

        # Mean squared error is the standard DDPM noise-prediction objective.
        loss = F.mse_loss(predicted_noise, noise)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=images.shape[0])
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Run one training batch.
        return self._shared_step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Run one validation batch.
        return self._shared_step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Run one test batch.
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # AdamW is a reliable default optimizer for diffusion UNets.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)


class ResourceMonitorCallback(pl.Callback):
    """Prints useful device and MPS memory information during training."""

    def __init__(self, log_every_n_steps: int) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Show which accelerator Lightning selected and where the model lives.
        root_device = trainer.strategy.root_device
        parameter_device = next(pl_module.parameters()).device
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Lightning root device: {root_device}")
        print(f"Model parameter device: {parameter_device}")

        # Warn if Apple GPU acceleration exists but the run is not using it.
        if torch.backends.mps.is_available() and root_device.type != "mps":
            print("Warning: MPS is available, but this run is not using the MPS device.")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        # Periodically print MPS memory usage without flooding the terminal.
        if self.log_every_n_steps <= 0:
            return
        if trainer.global_step == 0 or trainer.global_step % self.log_every_n_steps != 0:
            return
        if trainer.strategy.root_device.type != "mps":
            return

        allocated = torch.mps.current_allocated_memory()
        message = f"MPS memory at step {trainer.global_step}: allocated={format_gib(allocated)}"
        if hasattr(torch.mps, "driver_allocated_memory"):
            message += f", driver_allocated={format_gib(torch.mps.driver_allocated_memory())}"
        print(message)


def parse_args() -> argparse.Namespace:
    # Collect the few training options that are useful to change from the shell.
    parser = argparse.ArgumentParser(description="Train a small grayscale DDPM UNet.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--log-resource-every-n-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default=get_default_accelerator())
    parser.add_argument("--devices", type=str, default="1")
    return parser.parse_args()


def main() -> None:
    # Make data splits and dataloader worker behavior reproducible.
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # Prepare the image loaders and the Lightning training module.
    print(
        "Training with "
        f"accelerator={args.accelerator}, devices={args.devices}, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"gradient_checkpointing={not args.disable_gradient_checkpointing}"
    )
    if args.limit_train_batches is not None or args.limit_val_batches is not None:
        print(
            "Batch limits: "
            f"train={args.limit_train_batches or 'all'}, "
            f"val={args.limit_val_batches or 'all'}"
        )
    data_module = ImageDataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    data_module.print_summary(args.max_epochs, args.accumulate_grad_batches)
    diffusion_module = DiffusionModule(
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        image_size=args.image_size,
    )

    # Save the best model checkpoint according to validation loss.
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="diffusion-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Train the model, then evaluate the best validation checkpoint on the test split.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[
            checkpoint_callback,
            ResourceMonitorCallback(args.log_resource_every_n_steps),
        ],
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=(
            args.limit_train_batches if args.limit_train_batches is not None else 1.0
        ),
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=50,
    )
    trainer.fit(diffusion_module, datamodule=data_module)
    trainer.test(diffusion_module, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
