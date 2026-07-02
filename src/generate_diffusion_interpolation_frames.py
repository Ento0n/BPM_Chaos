from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler

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


def ease_progress(t: float, easing: str) -> float:
    if easing == "linear":
        return t
    if easing == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    raise ValueError(f"Unsupported easing: {easing}")


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
) -> list[tuple[int, int, float, int | None]]:
    frame_specs: list[tuple[int, int, float, int | None]] = []
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


def make_frame_noise_batch(
    anchor_noises: torch.Tensor,
    frame_specs: list[tuple[int, int, float, int | None]],
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
    checkpoint_path: Path,
    args: argparse.Namespace,
    segment_frames: int,
    frame_count: int,
    beat_count: int,
) -> None:
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
        "checkpoint": str(checkpoint_path),
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


def generate_diffusion_interpolation_frames(
    checkpoint_path: Path,
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
) -> tuple[int, int, torch.device, torch.device]:
    device = resolve_device(accelerator, required_device)

    print(f"Using {device} as device!")

    diffusion_module = DiffusionModule.load_from_checkpoint(
        checkpoint_path,
        learning_rate=1e-4,
        use_gradient_checkpointing=False,
        image_size=image_size,
        map_location=device,
    )

    print(f"Using checkpoint from {checkpoint_path}!")

    diffusion_module.eval()
    diffusion_module.to(device)
    parameter_device = validate_model_device(diffusion_module, device)

    scheduler = create_scheduler(scheduler_name, num_train_timesteps)
    set_scheduler_timesteps(scheduler, num_inference_steps, device)

    anchor_noises = generate_anchor_noises(num_beats, image_size, seed)
    frame_specs = build_frame_specs(num_beats, segment_frames, loop)
    generated_frame_count = len(frame_specs)
    frame_count = generated_frame_count
    beat_count = 0

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
        for batch_start in range(0, generated_frame_count, batch_size):
            batch_specs = frame_specs[batch_start : batch_start + batch_size]
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

            for batch_offset, image_tensor in enumerate(batch_images):
                frame_index = batch_start + batch_offset
                output_path = output_dir / f"frame_{frame_index:06d}.png"
                save_image(image_tensor, output_path)

                beat_index = batch_specs[batch_offset][3]
                if beat_index is not None and beat_output_dir is not None:
                    beat_path = beat_output_dir / f"sample_{beat_index:03d}.png"
                    save_image(image_tensor, beat_path)
                    beat_count += 1
    finally:
        if progress is not None:
            progress.close()

    if loop:
        closing_frame_path = output_dir / f"frame_{frame_count:06d}.png"
        shutil.copy2(output_dir / "frame_000000.png", closing_frame_path)
        frame_count += 1

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate beat and in-between frames by interpolating diffusion "
            "noise anchors before denoising."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        choices=("linear", "cosine"),
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
    return parser.parse_args()


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

    segment_frames = args.frames_per_beat or frames_per_beat(args.fps, args.bpm)
    if segment_frames <= 0:
        raise ValueError("--frames-per-beat must be greater than 0.")

    args.require_device = normalize_required_device(args.require_device, args.require_mps)
    checkpoint_path = args.checkpoint or find_checkpoint(args.checkpoint_dir)
    try:
        resolve_device(args.accelerator, args.require_device)
    except DeviceSelectionError as error:
        raise SystemExit(f"Error: {error}") from error

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
        args=args,
        segment_frames=segment_frames,
        frame_count=frame_count,
        beat_count=args.num_beats,
    )

    duration_seconds = frame_count / args.fps
    print(f"Run id: {run_id}")
    print(f"Run folder: {run_dir}")
    print(f"Loaded checkpoint: {checkpoint_path}")
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
