"""Generate diffusion frames from interpolated noise and optional model blends.

The default path loads one fixed diffusion checkpoint and behaves as before.
The opt-in ``random-binary`` model-alpha mode assigns alpha 0.00 or 1.00 to
each beat, linearly schedules the in-between frames at 0.01 resolution, groups
frames by alpha, and loads only one model checkpoint at a time for generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler

from easing import EASING_CHOICES, ease_progress
from generated_paths import (
    DEFAULT_DIFFUSION_BEAT_SUBDIR,
    DEFAULT_FRAME_SUBDIR,
    DEFAULT_GENERATED_DIR,
    create_unique_run_dir,
    validate_relative_subdir,
)
from generate_diffusion_images import find_checkpoint, get_device, save_image
from train_diffusion_model import (
    DEFAULT_OUTPUT_DIR,
    DiffusionModule,
    get_default_accelerator,
)

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ALPHA_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mixed_diffusion"
DEFAULT_MODEL_ALPHA_CHECKPOINT_PATTERN = (
    "centered_geometric_wood_diagonals_alpha_{alpha:.2f}.ckpt"
)
MODEL_ALPHA_MODES = ("fixed", "random-binary")
MODEL_ALPHA_QUANTIZATION = 0.01
FrameSpec = tuple[int, int, float, int | None]


@dataclass(frozen=True)
class ModelAlphaPlan:
    """Reproducible per-beat/per-frame model choices and resolved checkpoints."""

    seed: int
    beat_percents: tuple[int, ...]
    frame_percents: tuple[int, ...]
    checkpoint_paths: dict[int, Path]


class DeviceSelectionError(RuntimeError):
    pass


def create_timestamped_run_dirs(
    run_parent_dir: Path,
    frame_subdir: str,
    beat_subdir: str | None,
) -> tuple[str, Path, Path, Path | None]:
    frame_subdir_path = validate_relative_subdir(frame_subdir, "--frame-subdir")
    beat_subdir_path = (
        validate_relative_subdir(beat_subdir, "--beat-subdir")
        if beat_subdir is not None
        else None
    )

    run_id, run_dir = create_unique_run_dir(run_parent_dir)
    output_dir = run_dir / frame_subdir_path
    beat_output_dir = run_dir / beat_subdir_path if beat_subdir_path is not None else None

    output_dir.mkdir(parents=True, exist_ok=False)
    if beat_output_dir is not None:
        beat_output_dir.mkdir(parents=True, exist_ok=False)

    return run_id, run_dir, output_dir, beat_output_dir


def format_duration(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes_total, seconds_part = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


class TextProgress:
    def __init__(self, total: int, description: str) -> None:
        self.total = max(1, total)
        self.description = description
        self.current = 0
        self.last_percent = -1
        print(f"{self.description}: 0/{self.total}")

    def update(self, amount: int) -> None:
        self.current = min(self.total, self.current + amount)
        percent = int(self.current * 100 / self.total)
        if percent >= self.last_percent + 5 or self.current == self.total:
            self.last_percent = percent
            print(f"{self.description}: {self.current}/{self.total} ({percent}%)")

    def close(self) -> None:
        if self.current < self.total:
            print(f"{self.description}: {self.current}/{self.total}")


def create_progress(total: int, description: str, show_progress: bool):
    if not show_progress:
        return None
    if tqdm is not None:
        return tqdm(total=total, desc=description, unit="step")
    return TextProgress(total=total, description=description)


def frames_per_beat(fps: float, bpm: float) -> int:
    if fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if bpm <= 0:
        raise ValueError("--bpm must be greater than 0.")
    return max(1, int(round(fps * 60.0 / bpm)))


def resolve_device(accelerator: str, required_device: str | None) -> torch.device:
    device = get_device(accelerator)
    requested = accelerator.lower()

    if requested in {"mps", "cuda", "gpu"} and device.type == "cpu":
        raise DeviceSelectionError(
            f"Requested --accelerator {accelerator}, but PyTorch resolved CPU. "
            "Check that the requested device is available."
        )

    if required_device is not None and device.type != required_device:
        raise DeviceSelectionError(
            f"Required device {required_device!r}, but resolved {device.type!r}. "
            "Use a different --accelerator or remove --require-device."
        )

    return device


def validate_model_device(
    diffusion_module: DiffusionModule,
    expected_device: torch.device,
) -> torch.device:
    parameter_device = next(diffusion_module.parameters()).device
    if parameter_device.type != expected_device.type:
        raise DeviceSelectionError(
            f"Model parameters are on {parameter_device}, but expected {expected_device}."
        )
    return parameter_device


def lerp(start: torch.Tensor, end: torch.Tensor, t: float) -> torch.Tensor:
    return start * (1.0 - t) + end * t


def slerp(start: torch.Tensor, end: torch.Tensor, t: float) -> torch.Tensor:
    start_norm = torch.linalg.vector_norm(start)
    end_norm = torch.linalg.vector_norm(end)
    if start_norm <= 0 or end_norm <= 0:
        return lerp(start, end, t)

    start_unit = start / start_norm
    end_unit = end / end_norm
    dot = torch.sum(start_unit * end_unit).clamp(-1.0, 1.0)

    if torch.abs(dot) > 0.9995:
        return lerp(start, end, t)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    direction = (
        torch.sin((1.0 - t) * theta) / sin_theta * start_unit
        + torch.sin(t * theta) / sin_theta * end_unit
    )
    radius = start_norm * (1.0 - t) + end_norm * t
    return direction * radius


def interpolate_noise(
    start: torch.Tensor,
    end: torch.Tensor,
    t: float,
    interpolation: str,
) -> torch.Tensor:
    if interpolation == "lerp":
        return lerp(start, end, t)
    if interpolation == "slerp":
        return slerp(start, end, t)
    raise ValueError(f"Unsupported interpolation: {interpolation}")


def create_scheduler(name: str, num_train_timesteps: int) -> DDIMScheduler | DDPMScheduler:
    if name == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps)
    if name == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps)
    raise ValueError(f"Unsupported scheduler: {name}")


def set_scheduler_timesteps(
    scheduler: DDIMScheduler | DDPMScheduler,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps)

    for name, value in vars(scheduler).items():
        if torch.is_tensor(value):
            setattr(scheduler, name, value.to(device))


def scheduler_step(
    scheduler: DDIMScheduler | DDPMScheduler,
    predicted_noise: torch.Tensor,
    timestep: torch.Tensor,
    images: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    try:
        return scheduler.step(
            predicted_noise,
            timestep,
            images,
            eta=eta,
        ).prev_sample
    except TypeError:
        return scheduler.step(predicted_noise, timestep, images).prev_sample


def denoise_batch(
    diffusion_module: DiffusionModule,
    scheduler: DDIMScheduler | DDPMScheduler,
    initial_noise: torch.Tensor,
    device: torch.device,
    eta: float,
    progress,
) -> torch.Tensor:
    images = initial_noise.to(device)
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            if torch.is_tensor(timestep):
                timestep = timestep.to(device)
            else:
                timestep = torch.tensor(timestep, device=device)
            timesteps = timestep.repeat(images.shape[0])
            predicted_noise = diffusion_module.model(images, timesteps).sample
            images = scheduler_step(
                scheduler=scheduler,
                predicted_noise=predicted_noise,
                timestep=timestep,
                images=images,
                eta=eta,
            )
            if progress is not None:
                progress.update(images.shape[0])
    return images


def generate_anchor_noises(
    num_beats: int,
    image_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (num_beats, 1, image_size, image_size),
        generator=generator,
        device="cpu",
    )


def build_frame_specs(
    num_beats: int,
    segment_frames: int,
    loop: bool,
) -> list[FrameSpec]:
    frame_specs: list[FrameSpec] = []
    segment_count = num_beats if loop else num_beats - 1

    for segment_index in range(segment_count):
        start_index = segment_index
        end_index = (segment_index + 1) % num_beats

        for step in range(segment_frames):
            beat_index = start_index if step == 0 else None
            t = step / segment_frames
            frame_specs.append((start_index, end_index, t, beat_index))

    if not loop:
        frame_specs.append((num_beats - 1, num_beats - 1, 0.0, num_beats - 1))

    return frame_specs


# =============================================================================
# Random per-beat model-alpha planning
# =============================================================================


def build_random_binary_beat_alpha_percents(
    num_beats: int,
    seed: int,
) -> tuple[int, ...]:
    """Choose an independent, reproducible alpha endpoint for every beat."""
    if num_beats < 1:
        raise ValueError("num_beats must be at least 1 for model-alpha planning")

    # A local RNG keeps model selection reproducible without changing Python's
    # global random state or the separate torch generator used for noise anchors.
    rng = random.Random(seed)
    return tuple(rng.choice((0, 100)) for _ in range(num_beats))


def build_frame_alpha_percents(
    frame_specs: list[FrameSpec],
    beat_percents: tuple[int, ...],
) -> tuple[int, ...]:
    """Linearly interpolate beat alphas and quantize to the nearest percent."""
    if not beat_percents:
        raise ValueError("At least one beat alpha is required")
    if any(percent not in {0, 100} for percent in beat_percents):
        raise ValueError("Random-binary beat alphas must be either 0 or 100")

    frame_percents: list[int] = []
    for start_index, end_index, raw_t, _ in frame_specs:
        if start_index >= len(beat_percents) or end_index >= len(beat_percents):
            raise ValueError("Frame specification refers to a missing beat alpha")

        # Model blending intentionally uses raw linear frame progress. Noise
        # interpolation still uses --easing independently in make_frame_noise_batch.
        start_percent = beat_percents[start_index]
        end_percent = beat_percents[end_index]
        raw_percent = start_percent + (end_percent - start_percent) * raw_t

        # Alpha values are non-negative, so floor(x + 0.5) implements explicit
        # half-up rounding without Python's banker-rounding behavior at ties.
        quantized_percent = int(math.floor(raw_percent + 0.5))
        frame_percents.append(min(100, max(0, quantized_percent)))

    return tuple(frame_percents)


def resolve_model_alpha_checkpoints(
    required_percents: tuple[int, ...],
    checkpoint_dir: Path,
    checkpoint_pattern: str,
) -> dict[int, Path]:
    """Resolve every required alpha checkpoint before any run output is created."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Model-alpha checkpoint directory does not exist: {checkpoint_dir}"
        )

    checkpoint_paths: dict[int, Path] = {}
    resolved_paths: set[Path] = set()
    for percent in sorted(set(required_percents)):
        if not 0 <= percent <= 100:
            raise ValueError(f"Model alpha percent must be between 0 and 100: {percent}")
        try:
            filename = checkpoint_pattern.format(
                alpha=percent / 100,
                percent=percent,
            )
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(
                "--model-alpha-checkpoint-pattern must support "
                "{alpha} and/or {percent} formatting"
            ) from error

        # Keep the pattern to one filename inside the selected directory. This
        # makes the preflight target obvious and prevents accidental traversal.
        if not filename or Path(filename).name != filename:
            raise ValueError(
                "--model-alpha-checkpoint-pattern must produce a single filename"
            )
        checkpoint_path = checkpoint_dir / filename
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for alpha={percent / 100:.2f}: {checkpoint_path}"
            )
        if checkpoint_path in resolved_paths:
            raise ValueError(
                "--model-alpha-checkpoint-pattern produced the same file for "
                "multiple alpha values"
            )
        checkpoint_paths[percent] = checkpoint_path
        resolved_paths.add(checkpoint_path)

    return checkpoint_paths


def build_random_binary_model_alpha_plan(
    num_beats: int,
    frame_specs: list[FrameSpec],
    seed: int,
    checkpoint_dir: Path,
    checkpoint_pattern: str,
) -> ModelAlphaPlan:
    """Build the complete random-binary schedule and preflight its checkpoints."""
    beat_percents = build_random_binary_beat_alpha_percents(num_beats, seed)
    frame_percents = build_frame_alpha_percents(frame_specs, beat_percents)
    checkpoint_paths = resolve_model_alpha_checkpoints(
        required_percents=frame_percents,
        checkpoint_dir=checkpoint_dir,
        checkpoint_pattern=checkpoint_pattern,
    )
    return ModelAlphaPlan(
        seed=seed,
        beat_percents=beat_percents,
        frame_percents=frame_percents,
        checkpoint_paths=checkpoint_paths,
    )


def group_frame_indices_by_alpha(
    frame_percents: tuple[int, ...],
) -> dict[int, list[int]]:
    """Group original frame indices so each alpha model is loaded only once."""
    grouped_indices: dict[int, list[int]] = {}
    for frame_index, percent in enumerate(frame_percents):
        grouped_indices.setdefault(percent, []).append(frame_index)
    return grouped_indices


def saved_frame_alpha_percents(
    model_alpha_plan: ModelAlphaPlan,
    loop: bool,
) -> tuple[int, ...]:
    """Return metadata alphas, including the copied loop-closing frame."""
    if loop:
        return model_alpha_plan.frame_percents + (model_alpha_plan.beat_percents[0],)
    return model_alpha_plan.frame_percents


def make_frame_noise_batch(
    anchor_noises: torch.Tensor,
    frame_specs: list[FrameSpec],
    interpolation: str,
    easing: str,
) -> torch.Tensor:
    noises = []
    for start_index, end_index, raw_t, _ in frame_specs:
        t = ease_progress(raw_t, easing)
        noises.append(
            interpolate_noise(
                anchor_noises[start_index],
                anchor_noises[end_index],
                t,
                interpolation,
            )
        )
    return torch.stack(noises, dim=0)


def save_metadata(
    checkpoint_path: Path | None,
    model_alpha_plan: ModelAlphaPlan | None,
    args: argparse.Namespace,
    segment_frames: int,
    frame_count: int,
    beat_count: int,
) -> None:
    saved_alpha_percents = (
        saved_frame_alpha_percents(model_alpha_plan, args.loop)
        if model_alpha_plan is not None
        else ()
    )
    metadata = {
        "run_id": args.run_id,
        "run_parent_dir": str(args.run_parent_dir),
        "run_dir": str(args.run_dir),
        "frame_subdir": args.frame_subdir,
        "beat_subdir": None if args.no_save_beats else args.beat_subdir,
        "run_started_at": args.run_started_at,
        "run_finished_at": args.run_finished_at,
        "generation_elapsed_seconds": args.generation_elapsed_seconds,
        "generation_elapsed": args.generation_elapsed,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "model_alpha_mode": args.model_alpha_mode,
        "model_alpha_seed": (
            model_alpha_plan.seed if model_alpha_plan is not None else None
        ),
        "model_alpha_beat_values": (
            [percent / 100 for percent in model_alpha_plan.beat_percents]
            if model_alpha_plan is not None
            else None
        ),
        "model_alpha_frame_values": (
            [percent / 100 for percent in saved_alpha_percents]
            if model_alpha_plan is not None
            else None
        ),
        "model_alpha_interpolation": (
            "linear" if model_alpha_plan is not None else None
        ),
        "model_alpha_uses_noise_easing": (
            False if model_alpha_plan is not None else None
        ),
        "model_alpha_checkpoint_step": (
            MODEL_ALPHA_QUANTIZATION if model_alpha_plan is not None else None
        ),
        "model_alpha_checkpoint_dir": (
            str(args.model_alpha_checkpoint_dir)
            if model_alpha_plan is not None
            else None
        ),
        "model_alpha_checkpoint_pattern": (
            args.model_alpha_checkpoint_pattern
            if model_alpha_plan is not None
            else None
        ),
        "model_alpha_checkpoints": (
            {
                f"{percent / 100:.2f}": str(path)
                for percent, path in model_alpha_plan.checkpoint_paths.items()
            }
            if model_alpha_plan is not None
            else None
        ),
        "output_parent_dir": str(args.run_parent_dir),
        "output_dir": str(args.output_dir),
        "beat_output_parent_dir": None if args.no_save_beats else str(args.run_parent_dir),
        "beat_output_dir": None if args.no_save_beats else str(args.beat_output_dir),
        "num_beats": beat_count,
        "frame_count": frame_count,
        "fps": args.fps,
        "bpm": args.bpm,
        "frames_per_beat": segment_frames,
        "image_size": args.image_size,
        "num_inference_steps": args.num_inference_steps,
        "num_train_timesteps": args.num_train_timesteps,
        "seed": args.seed,
        "accelerator": args.accelerator,
        "require_device": args.require_device,
        "resolved_device": args.resolved_device,
        "scheduler": args.scheduler,
        "ddim_eta": args.ddim_eta,
        "interpolation": args.interpolation,
        "easing": args.easing,
        "loop": args.loop,
    }
    metadata_path = args.run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_diffusion_module(
    checkpoint_path: Path,
    image_size: int,
    device: torch.device,
) -> tuple[DiffusionModule, torch.device]:
    """Load one checkpoint on CPU, then move only that model to the target device."""
    diffusion_module = DiffusionModule.load_from_checkpoint(
        checkpoint_path,
        learning_rate=1e-4,
        use_gradient_checkpointing=False,
        image_size=image_size,
        map_location=torch.device("cpu"),
    )
    diffusion_module.eval()
    diffusion_module.to(device)
    parameter_device = validate_model_device(diffusion_module, device)
    return diffusion_module, parameter_device


def clear_device_cache(device: torch.device) -> None:
    """Release cached accelerator memory before loading the next alpha model."""
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def generate_indexed_frame_batches(
    frame_indices: list[int],
    frame_specs: list[FrameSpec],
    anchor_noises: torch.Tensor,
    diffusion_module: DiffusionModule,
    scheduler: DDIMScheduler | DDPMScheduler,
    output_dir: Path,
    beat_output_dir: Path | None,
    interpolation: str,
    easing: str,
    batch_size: int,
    device: torch.device,
    ddim_eta: float,
    progress,
) -> int:
    """Generate arbitrary timeline indices and save them under original filenames."""
    beat_count = 0
    for batch_start in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[batch_start : batch_start + batch_size]
        batch_specs = [frame_specs[index] for index in batch_indices]
        batch_noise = make_frame_noise_batch(
            anchor_noises=anchor_noises,
            frame_specs=batch_specs,
            interpolation=interpolation,
            easing=easing,
        )
        batch_images = denoise_batch(
            diffusion_module=diffusion_module,
            scheduler=scheduler,
            initial_noise=batch_noise,
            device=device,
            eta=ddim_eta,
            progress=progress,
        )

        # Generation may be grouped by model alpha rather than timeline order.
        # Original indices keep filenames sortable into the intended animation.
        for batch_offset, image_tensor in enumerate(batch_images):
            frame_index = batch_indices[batch_offset]
            output_path = output_dir / f"frame_{frame_index:06d}.png"
            save_image(image_tensor, output_path)

            beat_index = batch_specs[batch_offset][3]
            if beat_index is not None and beat_output_dir is not None:
                beat_path = beat_output_dir / f"sample_{beat_index:03d}.png"
                save_image(image_tensor, beat_path)
                beat_count += 1

        del batch_images, batch_noise

    return beat_count


def generate_diffusion_interpolation_frames(
    checkpoint_path: Path | None,
    output_dir: Path,
    beat_output_dir: Path | None,
    image_size: int,
    num_beats: int,
    segment_frames: int,
    num_inference_steps: int,
    num_train_timesteps: int,
    seed: int,
    accelerator: str,
    scheduler_name: str,
    ddim_eta: float,
    interpolation: str,
    easing: str,
    batch_size: int,
    loop: bool,
    required_device: str | None,
    show_progress: bool,
    model_alpha_plan: ModelAlphaPlan | None = None,
) -> tuple[int, int, torch.device, torch.device]:
    if (checkpoint_path is None) == (model_alpha_plan is None):
        raise ValueError(
            "Generation requires either one fixed checkpoint or one model-alpha plan"
        )

    device = resolve_device(accelerator, required_device)
    print(f"Using {device} as device!")

    # Anchor noise uses its own CPU generator below. Seed torch's global device
    # generators too so DDPM variance noise and DDIM eta > 0 remain repeatable
    # for the same frame grouping, batch size, and command-line seed.
    torch.manual_seed(seed)
    scheduler = create_scheduler(scheduler_name, num_train_timesteps)
    set_scheduler_timesteps(scheduler, num_inference_steps, device)

    anchor_noises = generate_anchor_noises(num_beats, image_size, seed)
    frame_specs = build_frame_specs(num_beats, segment_frames, loop)
    generated_frame_count = len(frame_specs)
    frame_count = generated_frame_count
    beat_count = 0
    parameter_device: torch.device | None = None

    if (
        model_alpha_plan is not None
        and len(model_alpha_plan.frame_percents) != generated_frame_count
    ):
        raise ValueError(
            "Model-alpha frame schedule length does not match generated frame count"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if beat_output_dir is not None:
        beat_output_dir.mkdir(parents=True, exist_ok=True)

    progress_total = generated_frame_count * len(scheduler.timesteps)
    progress = create_progress(
        total=progress_total,
        description=f"Denoising on {device.type}",
        show_progress=show_progress,
    )

    try:
        if model_alpha_plan is None:
            assert checkpoint_path is not None
            print(f"Using checkpoint from {checkpoint_path}!")
            diffusion_module, parameter_device = load_diffusion_module(
                checkpoint_path=checkpoint_path,
                image_size=image_size,
                device=device,
            )
            try:
                beat_count += generate_indexed_frame_batches(
                    frame_indices=list(range(generated_frame_count)),
                    frame_specs=frame_specs,
                    anchor_noises=anchor_noises,
                    diffusion_module=diffusion_module,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    beat_output_dir=beat_output_dir,
                    interpolation=interpolation,
                    easing=easing,
                    batch_size=batch_size,
                    device=device,
                    ddim_eta=ddim_eta,
                    progress=progress,
                )
            finally:
                diffusion_module.to(torch.device("cpu"))
                del diffusion_module
                clear_device_cache(device)
        else:
            grouped_indices = group_frame_indices_by_alpha(
                model_alpha_plan.frame_percents
            )
            for percent in sorted(grouped_indices):
                alpha_checkpoint_path = model_alpha_plan.checkpoint_paths[percent]
                frame_indices = grouped_indices[percent]
                print(
                    f"Using alpha={percent / 100:.2f} checkpoint "
                    f"for {len(frame_indices)} frame(s): {alpha_checkpoint_path}"
                )
                diffusion_module, parameter_device = load_diffusion_module(
                    checkpoint_path=alpha_checkpoint_path,
                    image_size=image_size,
                    device=device,
                )
                try:
                    beat_count += generate_indexed_frame_batches(
                        frame_indices=frame_indices,
                        frame_specs=frame_specs,
                        anchor_noises=anchor_noises,
                        diffusion_module=diffusion_module,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        beat_output_dir=beat_output_dir,
                        interpolation=interpolation,
                        easing=easing,
                        batch_size=batch_size,
                        device=device,
                        ddim_eta=ddim_eta,
                        progress=progress,
                    )
                finally:
                    diffusion_module.to(torch.device("cpu"))
                    del diffusion_module
                    clear_device_cache(device)
    finally:
        if progress is not None:
            progress.close()

    if loop:
        closing_frame_path = output_dir / f"frame_{frame_count:06d}.png"
        shutil.copy2(output_dir / "frame_000000.png", closing_frame_path)
        frame_count += 1

    if parameter_device is None:
        raise RuntimeError("No diffusion model was loaded for generation")
    return frame_count, beat_count, device, parameter_device


def normalize_required_device(
    require_device: str | None,
    require_mps: bool,
) -> str | None:
    if require_mps:
        if require_device is not None and require_device != "mps":
            raise ValueError("--require-mps conflicts with --require-device.")
        return "mps"
    return require_device


def validate_model_alpha_options(args: argparse.Namespace) -> None:
    """Reject option combinations whose checkpoint source would be ambiguous."""
    if args.model_alpha_mode == "random-binary" and args.checkpoint is not None:
        raise ValueError(
            "--checkpoint cannot be used with --model-alpha-mode random-binary; "
            "models come from --model-alpha-checkpoint-dir"
        )
    if args.model_alpha_mode == "fixed" and args.model_alpha_seed is not None:
        raise ValueError(
            "--model-alpha-seed requires --model-alpha-mode random-binary"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate beat and in-between frames by interpolating diffusion "
            "noise anchors before denoising."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-alpha-mode",
        choices=MODEL_ALPHA_MODES,
        default="fixed",
        help=(
            "fixed uses one --checkpoint as before; random-binary assigns "
            "alpha 0.00 or 1.00 to each beat and blends models between beats."
        ),
    )
    parser.add_argument(
        "--model-alpha-seed",
        type=int,
        default=None,
        help=(
            "Random beat-model seed for random-binary mode. "
            "Defaults to --seed."
        ),
    )
    parser.add_argument(
        "--model-alpha-checkpoint-dir",
        type=Path,
        default=DEFAULT_MODEL_ALPHA_CHECKPOINT_DIR,
        help="Directory containing the alpha checkpoints used by random-binary mode.",
    )
    parser.add_argument(
        "--model-alpha-checkpoint-pattern",
        type=str,
        default=DEFAULT_MODEL_ALPHA_CHECKPOINT_PATTERN,
        help=(
            "Filename format inside --model-alpha-checkpoint-dir. "
            "Available fields: {alpha} in [0,1] and {percent} in [0,100]."
        ),
    )
    parser.add_argument(
        "--run-parent-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Parent directory for timestamped run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --run-parent-dir.",
    )
    parser.add_argument(
        "--frame-subdir",
        type=str,
        default=DEFAULT_FRAME_SUBDIR,
        help="Subdirectory inside the timestamped run folder for frame_*.png.",
    )
    parser.add_argument(
        "--beat-subdir",
        type=str,
        default=DEFAULT_DIFFUSION_BEAT_SUBDIR,
        help="Subdirectory inside the timestamped run folder for sample_*.png beat images.",
    )
    parser.add_argument(
        "--beat-output-dir",
        type=Path,
        default=None,
        help=(
            "Deprecated alias for --beat-subdir. If supplied, only the final "
            "path component is used."
        ),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-beats", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--frames-per-beat", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default=get_default_accelerator())
    parser.add_argument(
        "--require-device",
        choices=("cpu", "mps", "cuda"),
        default=None,
        help="Fail unless the resolved torch device has this type.",
    )
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Shortcut for --require-device mps.",
    )
    parser.add_argument("--scheduler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity. Keep 0 for deterministic interpolation.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("slerp", "lerp"),
        default="slerp",
        help="How to interpolate between diffusion noise anchors.",
    )
    parser.add_argument(
        "--easing",
        choices=EASING_CHOICES,
        default="cosine",
        help="Progress curve between beat anchors.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Interpolate from the last beat anchor back to the first, then reuse "
            "the first beat image as the final frame."
        ),
    )
    parser.add_argument(
        "--no-save-beats",
        action="store_true",
        help="Only save frame_*.png and skip the sample_*.png beat directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Kept for command compatibility; timestamped run folders avoid overwrites.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the denoising progress bar.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_started_at = datetime.now()
    run_timer_started = time.perf_counter()

    if args.num_beats < 2:
        raise ValueError("--num-beats must be at least 2.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be greater than 0.")
    if args.num_train_timesteps <= 0:
        raise ValueError("--num-train-timesteps must be greater than 0.")
    validate_model_alpha_options(args)

    segment_frames = args.frames_per_beat or frames_per_beat(args.fps, args.bpm)
    if segment_frames <= 0:
        raise ValueError("--frames-per-beat must be greater than 0.")

    args.require_device = normalize_required_device(args.require_device, args.require_mps)
    try:
        resolve_device(args.accelerator, args.require_device)
    except DeviceSelectionError as error:
        raise SystemExit(f"Error: {error}") from error

    # Resolve every required input before creating a timestamped output folder.
    # A missing alpha checkpoint therefore fails without leaving a partial run.
    model_alpha_plan: ModelAlphaPlan | None = None
    checkpoint_path: Path | None = None
    if args.model_alpha_mode == "fixed":
        checkpoint_path = args.checkpoint or find_checkpoint(args.checkpoint_dir)
    else:
        model_alpha_seed = (
            args.model_alpha_seed
            if args.model_alpha_seed is not None
            else args.seed
        )
        args.model_alpha_checkpoint_dir = (
            args.model_alpha_checkpoint_dir.expanduser().resolve()
        )
        planned_frame_specs = build_frame_specs(
            num_beats=args.num_beats,
            segment_frames=segment_frames,
            loop=args.loop,
        )
        model_alpha_plan = build_random_binary_model_alpha_plan(
            num_beats=args.num_beats,
            frame_specs=planned_frame_specs,
            seed=model_alpha_seed,
            checkpoint_dir=args.model_alpha_checkpoint_dir,
            checkpoint_pattern=args.model_alpha_checkpoint_pattern,
        )

    run_parent_dir = args.output_dir or args.run_parent_dir
    beat_subdir = None if args.no_save_beats else args.beat_subdir
    if args.beat_output_dir is not None and beat_subdir is not None:
        beat_subdir = args.beat_output_dir.name

    run_id, run_dir, output_dir, beat_output_dir = create_timestamped_run_dirs(
        run_parent_dir=run_parent_dir,
        frame_subdir=args.frame_subdir,
        beat_subdir=beat_subdir,
    )

    args.run_id = run_id
    args.run_parent_dir = run_parent_dir
    args.run_dir = run_dir
    args.beat_subdir = beat_subdir
    args.output_dir = output_dir
    args.beat_output_dir = beat_output_dir

    try:
        frame_count, beat_count, resolved_device, parameter_device = (
            generate_diffusion_interpolation_frames(
                checkpoint_path=checkpoint_path,
                model_alpha_plan=model_alpha_plan,
                output_dir=args.output_dir,
                beat_output_dir=beat_output_dir,
                image_size=args.image_size,
                num_beats=args.num_beats,
                segment_frames=segment_frames,
                num_inference_steps=args.num_inference_steps,
                num_train_timesteps=args.num_train_timesteps,
                seed=args.seed,
                accelerator=args.accelerator,
                scheduler_name=args.scheduler,
                ddim_eta=args.ddim_eta,
                interpolation=args.interpolation,
                easing=args.easing,
                batch_size=args.batch_size,
                loop=args.loop,
                required_device=args.require_device,
                show_progress=not args.no_progress,
            )
        )
    except DeviceSelectionError as error:
        raise SystemExit(f"Error: {error}") from error
    args.resolved_device = str(resolved_device)

    run_finished_at = datetime.now()
    elapsed_seconds = time.perf_counter() - run_timer_started
    elapsed_text = format_duration(elapsed_seconds)
    args.run_started_at = run_started_at.isoformat(timespec="seconds")
    args.run_finished_at = run_finished_at.isoformat(timespec="seconds")
    args.generation_elapsed_seconds = round(elapsed_seconds, 3)
    args.generation_elapsed = elapsed_text

    save_metadata(
        checkpoint_path=checkpoint_path,
        model_alpha_plan=model_alpha_plan,
        args=args,
        segment_frames=segment_frames,
        frame_count=frame_count,
        beat_count=args.num_beats,
    )

    duration_seconds = frame_count / args.fps
    print(f"Run id: {run_id}")
    print(f"Run folder: {run_dir}")
    if model_alpha_plan is None:
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        beat_alpha_text = ", ".join(
            f"{percent / 100:.2f}" for percent in model_alpha_plan.beat_percents
        )
        print("Model alpha mode: random-binary")
        print(f"Model alpha seed: {model_alpha_plan.seed}")
        print(f"Beat model alphas: {beat_alpha_text}")
        print(
            "Unique model alpha checkpoints used: "
            f"{len(model_alpha_plan.checkpoint_paths)}"
        )
    print(f"Requested accelerator: {args.accelerator}")
    print(f"Resolved torch device: {resolved_device}")
    print(f"Model parameter device: {parameter_device}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Interpolation: {args.interpolation}")
    print(f"Easing: {args.easing}")
    print(f"Beat anchors: {args.num_beats}")
    print(f"Frames per beat: {segment_frames}")
    print(f"Saved {frame_count} frames to {args.output_dir}")
    if beat_output_dir is not None:
        print(f"Saved {beat_count} beat images to {beat_output_dir}")
    print(f"Approx. duration at {args.fps:g} fps: {duration_seconds:.2f}s")
    print(f"Generation elapsed: {elapsed_text} ({elapsed_seconds:.2f}s)")


if __name__ == "__main__":
    main()
