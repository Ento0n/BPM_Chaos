from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from generated_paths import (
    DEFAULT_DIFFUSION_BEAT_SUBDIR,
    DEFAULT_FRAME_SUBDIR,
    DEFAULT_VIDEO_NAME,
    DEFAULT_VIDEO_SUBDIR,
    validate_relative_subdir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEAT_DIR = PROJECT_ROOT / "generated" / "diffusion"
DEFAULT_FRAME_DIR = PROJECT_ROOT / "generated" / "crossfade"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "generated" / "videos" / "crossfade.mp4"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
DEFAULT_COLOR_0 = (0, 0, 0)
DEFAULT_COLOR_1 = (255, 255, 255)
RGBColor = tuple[int, int, int]
ColorMap = tuple[RGBColor, RGBColor]


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_image_paths(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        (
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def parse_hex_color(value: str) -> RGBColor:
    color = value.strip()
    if color.startswith("#"):
        color = color[1:]

    if len(color) == 3:
        color = "".join(channel * 2 for channel in color)

    if not re.fullmatch(r"[0-9a-fA-F]{6}", color):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid hex color. Use RRGGBB or #RRGGBB."
        )

    return tuple(int(color[index : index + 2], 16) for index in range(0, 6, 2))


def format_hex_color(color: RGBColor) -> str:
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def frames_per_beat(fps: float, bpm: float) -> int:
    if fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if bpm <= 0:
        raise ValueError("--bpm must be greater than 0.")
    return max(1, int(round(fps * 60.0 / bpm)))


def build_frame_sequence(
    frame_paths: list[Path],
    beat_paths: list[Path],
    segment_frames: int,
    loop: bool,
) -> tuple[list[Path], str]:
    if len(frame_paths) < 1:
        raise ValueError("No rendered frames found.")

    if len(beat_paths) < 2:
        return frame_paths, "Using frame directory as a complete ordered sequence."

    segment_count = len(beat_paths) if loop else len(beat_paths) - 1
    complete_count = segment_count * segment_frames + 1
    open_loop_count = segment_count * segment_frames if loop else None
    intermediate_count = segment_count * max(0, segment_frames - 1)

    if len(frame_paths) == complete_count:
        return (
            frame_paths,
            "Frame directory already contains the complete closed sequence including beat images.",
        )

    if open_loop_count is not None and len(frame_paths) == open_loop_count:
        return (
            frame_paths + [frame_paths[0]],
            "Frame directory contains an open loop sequence; reused the first frame as the closing frame.",
        )

    if len(frame_paths) == intermediate_count:
        sequence: list[Path] = []
        frame_index = 0
        transition_count = max(0, segment_frames - 1)

        for segment_index in range(segment_count):
            sequence.append(beat_paths[segment_index])
            sequence.extend(frame_paths[frame_index : frame_index + transition_count])
            frame_index += transition_count

        sequence.append(beat_paths[0] if loop else beat_paths[-1])

        return sequence, "Merged beat images with intermediate transition frames."

    complete_count_message = str(complete_count)
    if open_loop_count is not None:
        complete_count_message = (
            f"{complete_count} complete frames or {open_loop_count} open-loop frames"
        )

    raise ValueError(
        "Frame count does not match the beat image count. "
        f"Found {len(beat_paths)} beat images and {len(frame_paths)} frame images. "
        f"Expected either {complete_count_message} or "
        f"{intermediate_count} intermediate frames for {segment_frames} frames per beat."
    )


def validate_frame_sizes(frame_paths: list[Path]) -> tuple[int, int]:
    with Image.open(frame_paths[0]) as first_image:
        expected_size = first_image.size

    for frame_path in frame_paths[1:]:
        with Image.open(frame_path) as image:
            if image.size != expected_size:
                raise ValueError(
                    f"Image size mismatch: {frame_path} is {image.size}, "
                    f"expected {expected_size}."
                )

    return expected_size


def symlink_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def color_channel_lookup(zero_value: int, one_value: int) -> list[int]:
    return [
        int(round(zero_value + (one_value - zero_value) * pixel / 255.0))
        for pixel in range(256)
    ]


def colorize_image(image: Image.Image, color_map: ColorMap) -> Image.Image:
    color_0, color_1 = color_map
    grayscale = image.convert("L")
    channels = [
        grayscale.point(color_channel_lookup(color_0[index], color_1[index]))
        for index in range(3)
    ]
    return Image.merge("RGB", channels)


def save_colorized_frame(source: Path, destination: Path, color_map: ColorMap) -> None:
    with Image.open(source) as image:
        colorized = colorize_image(image, color_map)

    try:
        colorized.save(destination)
    finally:
        colorized.close()


def create_ffmpeg_sequence(
    frame_paths: list[Path],
    sequence_dir: Path,
    color_map: ColorMap | None,
) -> str:
    output_pattern = "frame_%06d.png"
    for index, frame_path in enumerate(frame_paths):
        output_path = sequence_dir / f"frame_{index:06d}.png"
        if color_map is None:
            symlink_or_copy(frame_path, output_path)
        else:
            save_colorized_frame(frame_path, output_path, color_map)
    return output_pattern


def render_video_with_ffmpeg(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    audio_path: Path | None,
    overwrite: bool,
    color_map: ColorMap | None,
) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required for MP4/MOV/MKV/WebM/AVI output, but it was not found. "
            "Use a .gif output for a dependency-light preview or install ffmpeg."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    with tempfile.TemporaryDirectory(prefix="bpm_video_frames_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_pattern = create_ffmpeg_sequence(frame_paths, temp_dir, color_map)
        command = [
            ffmpeg_path,
            "-y" if overwrite else "-n",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(temp_dir / input_pattern),
        ]

        if audio_path is not None:
            command.extend(["-i", str(audio_path), "-shortest"])

        command.extend(
            [
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ]
        )

        if audio_path is not None:
            command.extend(["-c:a", "aac", "-b:a", "192k"])

        command.append(str(output_path))
        subprocess.run(command, check=True)


def load_gif_frame(frame_path: Path, color_map: ColorMap | None) -> Image.Image:
    with Image.open(frame_path) as image:
        if color_map is None:
            return image.convert("P", palette=Image.Palette.ADAPTIVE)

        colorized = colorize_image(image, color_map)

    try:
        return colorized.convert("P", palette=Image.Palette.ADAPTIVE)
    finally:
        colorized.close()


def render_gif(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    overwrite: bool,
    color_map: ColorMap | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    duration_ms = max(1, int(round(1000.0 / fps)))
    frames = [load_gif_frame(frame_path, color_map) for frame_path in frame_paths]
    first_frame = frames[0]
    remaining_frames = frames[1:]

    first_frame.save(
        output_path,
        save_all=True,
        append_images=remaining_frames,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    for frame in frames:
        frame.close()


def render_output(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    audio_path: Path | None,
    overwrite: bool,
    color_map: ColorMap | None,
) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        if audio_path is not None:
            raise ValueError("GIF output cannot include audio. Use an MP4 output instead.")
        render_gif(frame_paths, output_path, fps, overwrite, color_map)
        return

    if suffix in VIDEO_EXTENSIONS:
        render_video_with_ffmpeg(
            frame_paths,
            output_path,
            fps,
            audio_path,
            overwrite,
            color_map,
        )
        return

    raise ValueError(
        f"Unsupported output extension {output_path.suffix!r}. "
        "Use .mp4 for video or .gif for a quick preview."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge generated beat images and transition frames into a video."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Timestamped generated run folder containing beat, frame, and video subdirectories.",
    )
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--beat-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--frame-subdir",
        type=str,
        default=DEFAULT_FRAME_SUBDIR,
        help="Subdirectory inside --run-dir containing frame_*.png.",
    )
    parser.add_argument(
        "--beat-subdir",
        type=str,
        default=DEFAULT_DIFFUSION_BEAT_SUBDIR,
        help="Subdirectory inside --run-dir containing beat images.",
    )
    parser.add_argument(
        "--video-subdir",
        type=str,
        default=DEFAULT_VIDEO_SUBDIR,
        help="Subdirectory inside --run-dir for the default output file.",
    )
    parser.add_argument("--frame-pattern", type=str, default="frame_*.png")
    parser.add_argument("--beat-pattern", type=str, default="*.png")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument(
        "--color-0",
        "--zero-color",
        dest="color_0",
        type=parse_hex_color,
        default=None,
        help=(
            "Hex color for binary value 0. Accepts RRGGBB or #RRGGBB. "
            "Defaults to #000000 when either color option is used."
        ),
    )
    parser.add_argument(
        "--color-1",
        "--one-color",
        dest="color_1",
        type=parse_hex_color,
        default=None,
        help=(
            "Hex color for binary value 1. Accepts RRGGBB or #RRGGBB. "
            "Defaults to #FFFFFF when either color option is used."
        ),
    )
    parser.add_argument(
        "--frames-per-beat",
        type=int,
        default=None,
        help="Defaults to round(fps * 60 / bpm).",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio file. Requires video output and ffmpeg.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Expect or assemble a closed loop from the last beat image back to "
            "the first, reusing the first beat image as the final frame."
        ),
    )
    parser.add_argument(
        "--skip-beat-validation",
        action="store_true",
        help="Use frame-dir directly without checking it against beat-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def resolve_run_paths(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        args.frame_dir = args.frame_dir or DEFAULT_FRAME_DIR
        args.beat_dir = args.beat_dir or DEFAULT_BEAT_DIR
        args.output = args.output or DEFAULT_OUTPUT_PATH
        return

    frame_subdir = validate_relative_subdir(args.frame_subdir, "--frame-subdir")
    beat_subdir = validate_relative_subdir(args.beat_subdir, "--beat-subdir")
    video_subdir = validate_relative_subdir(args.video_subdir, "--video-subdir")
    args.frame_dir = args.frame_dir or args.run_dir / frame_subdir
    args.beat_dir = args.beat_dir or args.run_dir / beat_subdir
    args.output = args.output or args.run_dir / video_subdir / DEFAULT_VIDEO_NAME


def resolve_color_map(args: argparse.Namespace) -> ColorMap | None:
    if args.color_0 is None and args.color_1 is None:
        return None

    return (
        args.color_0 or DEFAULT_COLOR_0,
        args.color_1 or DEFAULT_COLOR_1,
    )


def main() -> None:
    args = parse_args()
    resolve_run_paths(args)
    segment_frames = args.frames_per_beat or frames_per_beat(args.fps, args.bpm)
    if segment_frames <= 0:
        raise ValueError("--frames-per-beat must be greater than 0.")

    frame_paths = collect_image_paths(args.frame_dir, args.frame_pattern)
    beat_paths = [] if args.skip_beat_validation else collect_image_paths(args.beat_dir, args.beat_pattern)

    sequence_paths, sequence_message = build_frame_sequence(
        frame_paths=frame_paths,
        beat_paths=beat_paths,
        segment_frames=segment_frames,
        loop=args.loop,
    )
    image_size = validate_frame_sizes(sequence_paths)

    audio_path = args.audio
    if audio_path is not None and not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    color_map = resolve_color_map(args)

    render_output(
        frame_paths=sequence_paths,
        output_path=args.output,
        fps=args.fps,
        audio_path=audio_path,
        overwrite=args.overwrite,
        color_map=color_map,
    )

    duration_seconds = len(sequence_paths) / args.fps
    if args.run_dir is not None:
        print(f"Run folder: {args.run_dir}")
    print(sequence_message)
    print(f"Frames per beat: {segment_frames}")
    print(f"Video frames: {len(sequence_paths)}")
    print(f"Frame size: {image_size[0]}x{image_size[1]}")
    if color_map is not None:
        print(f"Color 0: {format_hex_color(color_map[0])}")
        print(f"Color 1: {format_hex_color(color_map[1])}")
    print(f"Duration at {args.fps:g} fps: {duration_seconds:.2f}s")
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
