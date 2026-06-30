from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "generated" / "diffusion"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated" / "crossfade"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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


def generate_crossfade_frames(
    image_paths: list[Path],
    output_dir: Path,
    mode: str,
    segment_frames: int,
    loop: bool,
    overwrite: bool,
) -> int:
    prepare_output_dir(output_dir, overwrite)

    first_image = load_image(image_paths[0], mode)
    image_size = first_image.size
    start_image = first_image
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

        for step in range(segment_frames):
            t = step / segment_frames
            output_path = output_dir / f"frame_{frame_index:06d}.png"
            Image.blend(start_image, end_image, t).save(output_path)
            frame_index += 1

        start_image = end_image

    if not loop:
        output_path = output_dir / f"frame_{frame_index:06d}.png"
        start_image.save(output_path)
        frame_index += 1

    return frame_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate linear pixel-crossfade frames between beat images."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", type=str, default="*.png")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bpm", type=float, default=120.0)
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
        help="Also crossfade from the last beat image back to the first.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing frame_*.png files in the output directory before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    )

    beat_count = len(image_paths)
    duration_seconds = frame_count / args.fps
    print(f"Loaded {beat_count} beat images from {args.input_dir}")
    print(f"Frames per beat: {segment_frames}")
    print(f"Saved {frame_count} frames to {args.output_dir}")
    print(f"Approx. duration at {args.fps:g} fps: {duration_seconds:.2f}s")


if __name__ == "__main__":
    main()
