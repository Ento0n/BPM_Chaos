from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from easing import EASING_CHOICES, ease_progress
from generated_paths import (
    DEFAULT_DIFFUSION_BEAT_SUBDIR,
    DEFAULT_FRAME_SUBDIR,
    validate_relative_subdir,
)

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "generated" / "diffusion"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated" / "crossfade"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SQRT_2 = 2**0.5


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort sample_2.png before sample_10.png."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_image_paths(input_dir: Path, pattern: str) -> list[Path]:
    image_paths = sorted(
        (
            path
            for path in input_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two images in {input_dir} matching {pattern!r}.")
    return image_paths


def frames_per_beat(fps: float, bpm: float) -> int:
    if fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if bpm <= 0:
        raise ValueError("--bpm must be greater than 0.")
    return max(1, int(round(fps * 60.0 / bpm)))


def load_image(path: Path, mode: str) -> Image.Image:
    return Image.open(path).convert(mode)


def require_numpy() -> None:
    if np is None:
        raise RuntimeError("--method sdf requires numpy, but numpy is not installed.")


@dataclass(frozen=True)
class SdfImage:
    signed_distance: np.ndarray
    foreground_value: float
    background_value: float


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("frame_*.png"), key=natural_sort_key)
    if not existing_frames:
        return
    if not overwrite:
        raise ValueError(
            f"{output_dir} already contains frame_*.png files. "
            "Use an empty output directory or pass --overwrite."
        )
    for frame_path in existing_frames:
        frame_path.unlink()


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    require_numpy()
    height, width = mask.shape
    max_distance = float(np.hypot(height, width))
    if not mask.any():
        return np.full(mask.shape, max_distance, dtype=np.float32)

    distances = np.where(mask, 0.0, max_distance).astype(np.float32)

    for y in range(height):
        for x in range(width):
            best = distances[y, x]
            if x > 0:
                best = min(best, distances[y, x - 1] + 1.0)
            if y > 0:
                best = min(best, distances[y - 1, x] + 1.0)
            if x > 0 and y > 0:
                best = min(best, distances[y - 1, x - 1] + SQRT_2)
            if x + 1 < width and y > 0:
                best = min(best, distances[y - 1, x + 1] + SQRT_2)
            distances[y, x] = best

    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            best = distances[y, x]
            if x + 1 < width:
                best = min(best, distances[y, x + 1] + 1.0)
            if y + 1 < height:
                best = min(best, distances[y + 1, x] + 1.0)
            if x + 1 < width and y + 1 < height:
                best = min(best, distances[y + 1, x + 1] + SQRT_2)
            if x > 0 and y + 1 < height:
                best = min(best, distances[y + 1, x - 1] + SQRT_2)
            distances[y, x] = best

    return distances


def foreground_mask(gray_pixels: np.ndarray, threshold: int, foreground: str) -> np.ndarray:
    white_mask = gray_pixels >= threshold
    if foreground == "white":
        return white_mask
    if foreground == "black":
        return ~white_mask
    if foreground == "auto":
        white_pixels = int(white_mask.sum())
        black_pixels = int(white_mask.size - white_pixels)
        return white_mask if white_pixels <= black_pixels else ~white_mask
    raise ValueError(f"Unsupported foreground setting: {foreground}")


def median_pixel_value(gray_pixels: np.ndarray, mask: np.ndarray, fallback: int) -> float:
    if not mask.any():
        return float(fallback)
    return float(np.median(gray_pixels[mask]))


def signed_distance_field(mask: np.ndarray) -> np.ndarray:
    distance_to_foreground = distance_to_mask(mask)
    distance_to_background = distance_to_mask(~mask)
    return distance_to_foreground - distance_to_background


def prepare_sdf_image(image: Image.Image, threshold: int, foreground: str) -> SdfImage:
    require_numpy()
    gray_pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = foreground_mask(gray_pixels, threshold, foreground)
    foreground_value = median_pixel_value(gray_pixels, mask, 255)
    background_value = median_pixel_value(gray_pixels, ~mask, 0)
    return SdfImage(
        signed_distance=signed_distance_field(mask),
        foreground_value=foreground_value,
        background_value=background_value,
    )


def render_sdf_frame(
    signed_distance: np.ndarray,
    foreground_value: float,
    background_value: float,
    edge_softness: float,
    mode: str,
) -> Image.Image:
    if edge_softness <= 0:
        alpha = signed_distance <= 0
        gray_pixels = np.where(alpha, foreground_value, background_value)
    else:
        alpha = np.clip(0.5 - signed_distance / edge_softness, 0.0, 1.0)
        gray_pixels = background_value * (1.0 - alpha) + foreground_value * alpha

    gray_pixels = np.clip(np.rint(gray_pixels), 0, 255).astype(np.uint8)
    image = Image.fromarray(gray_pixels)
    if mode == "L":
        return image
    if mode == "RGB":
        return image.convert("RGB")
    if mode == "RGBA":
        return image.convert("RGBA")
    raise ValueError(f"Unsupported image mode: {mode}")


def blend_sdf_images(
    start_image: SdfImage,
    end_image: SdfImage,
    t: float,
    edge_softness: float,
    mode: str,
) -> Image.Image:
    signed_distance = (
        start_image.signed_distance * (1.0 - t)
        + end_image.signed_distance * t
    )
    foreground_value = (
        start_image.foreground_value * (1.0 - t)
        + end_image.foreground_value * t
    )
    background_value = (
        start_image.background_value * (1.0 - t)
        + end_image.background_value * t
    )
    return render_sdf_frame(
        signed_distance=signed_distance,
        foreground_value=foreground_value,
        background_value=background_value,
        edge_softness=edge_softness,
        mode=mode,
    )


def generate_crossfade_frames(
    image_paths: list[Path],
    output_dir: Path,
    mode: str,
    segment_frames: int,
    loop: bool,
    overwrite: bool,
    method: str,
    easing: str,
    threshold: int,
    foreground: str,
    sdf_edge_softness: float,
) -> int:
    prepare_output_dir(output_dir, overwrite)

    first_image = load_image(image_paths[0], mode)
    image_size = first_image.size
    start_image = first_image
    start_sdf_image = (
        prepare_sdf_image(start_image, threshold, foreground)
        if method == "sdf"
        else None
    )
    frame_index = 0

    target_paths = image_paths[1:]
    if loop:
        target_paths = target_paths + [image_paths[0]]

    for target_path in target_paths:
        end_image = load_image(target_path, mode)
        if end_image.size != image_size:
            raise ValueError(
                f"Image size mismatch: {target_path} is {end_image.size}, "
                f"expected {image_size}."
            )
        end_sdf_image = (
            prepare_sdf_image(end_image, threshold, foreground)
            if method == "sdf"
            else None
        )

        for step in range(segment_frames):
            output_path = output_dir / f"frame_{frame_index:06d}.png"
            if step == 0:
                start_image.save(output_path)
            else:
                t = ease_progress(step / segment_frames, easing)
                if method == "pixel":
                    Image.blend(start_image, end_image, t).save(output_path)
                elif method == "sdf":
                    if start_sdf_image is None or end_sdf_image is None:
                        raise RuntimeError("SDF images were not prepared.")
                    blend_sdf_images(
                        start_image=start_sdf_image,
                        end_image=end_sdf_image,
                        t=t,
                        edge_softness=sdf_edge_softness,
                        mode=mode,
                    ).save(output_path)
                else:
                    raise ValueError(f"Unsupported method: {method}")
            frame_index += 1

        start_image = end_image
        start_sdf_image = end_sdf_image

    if loop:
        output_path = output_dir / f"frame_{frame_index:06d}.png"
        shutil.copy2(output_dir / "frame_000000.png", output_path)
        frame_index += 1
    else:
        output_path = output_dir / f"frame_{frame_index:06d}.png"
        start_image.save(output_path)
        frame_index += 1

    return frame_index


def save_metadata(args: argparse.Namespace, segment_frames: int, frame_count: int) -> None:
    metadata = {
        "run_dir": None if args.run_dir is None else str(args.run_dir),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_subdir": args.input_subdir,
        "output_subdir": args.output_subdir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frame_count": frame_count,
        "fps": args.fps,
        "bpm": args.bpm,
        "frames_per_beat": segment_frames,
        "method": args.method,
        "easing": args.easing,
        "loop": args.loop,
        "mode": args.mode,
        "threshold": args.threshold,
        "foreground": args.foreground,
        "sdf_edge_softness": args.sdf_edge_softness,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate linear pixel-crossfade frames between beat images."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Timestamped generated run folder containing beat and frame subdirectories.",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--input-subdir",
        type=str,
        default=DEFAULT_DIFFUSION_BEAT_SUBDIR,
        help="Subdirectory inside --run-dir containing beat images.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_FRAME_SUBDIR,
        help="Subdirectory inside --run-dir for generated frame_*.png files.",
    )
    parser.add_argument("--pattern", type=str, default="*.png")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument(
        "--method",
        choices=("pixel", "sdf"),
        default="pixel",
        help="Interpolation method used between beat images.",
    )
    parser.add_argument(
        "--easing",
        choices=EASING_CHOICES,
        default="linear",
        help="Progress curve used inside each beat-to-beat transition.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Grayscale threshold used to build SDF masks.",
    )
    parser.add_argument(
        "--foreground",
        choices=("auto", "white", "black"),
        default="auto",
        help="Which side of the threshold should be treated as the shape.",
    )
    parser.add_argument(
        "--sdf-edge-softness",
        type=float,
        default=1.5,
        help="Width of the anti-aliased SDF edge in pixels. Use 0 for hard edges.",
    )
    parser.add_argument(
        "--frames-per-beat",
        type=int,
        default=None,
        help=(
            "Video frames from one beat image to the next. "
            "Defaults to round(fps * 60 / bpm)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("L", "RGB", "RGBA"),
        default="L",
        help="PIL image mode used for blending and output.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Crossfade from the last beat image back to the first, then reuse "
            "the first beat image as the final frame."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing frame_*.png files in the output directory before writing.",
    )
    return parser.parse_args()


def resolve_run_paths(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        args.input_dir = args.input_dir or DEFAULT_INPUT_DIR
        args.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        return

    input_subdir = validate_relative_subdir(args.input_subdir, "--input-subdir")
    output_subdir = validate_relative_subdir(args.output_subdir, "--output-subdir")
    args.input_dir = args.input_dir or args.run_dir / input_subdir
    args.output_dir = args.output_dir or args.run_dir / output_subdir


def main() -> None:
    args = parse_args()
    resolve_run_paths(args)
    segment_frames = args.frames_per_beat or frames_per_beat(args.fps, args.bpm)
    if segment_frames <= 0:
        raise ValueError("--frames-per-beat must be greater than 0.")

    image_paths = collect_image_paths(args.input_dir, args.pattern)
    frame_count = generate_crossfade_frames(
        image_paths=image_paths,
        output_dir=args.output_dir,
        mode=args.mode,
        segment_frames=segment_frames,
        loop=args.loop,
        overwrite=args.overwrite,
        method=args.method,
        easing=args.easing,
        threshold=args.threshold,
        foreground=args.foreground,
        sdf_edge_softness=args.sdf_edge_softness,
    )
    save_metadata(args, segment_frames, frame_count)

    beat_count = len(image_paths)
    duration_seconds = frame_count / args.fps
    if args.run_dir is not None:
        print(f"Run folder: {args.run_dir}")
    print(f"Loaded {beat_count} beat images from {args.input_dir}")
    print(f"Interpolation method: {args.method}")
    print(f"Easing: {args.easing}")
    print(f"Frames per beat: {segment_frames}")
    print(f"Saved {frame_count} frames to {args.output_dir}")
    print(f"Approx. duration at {args.fps:g} fps: {duration_seconds:.2f}s")


if __name__ == "__main__":
    main()
