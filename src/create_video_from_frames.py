"""Assemble generated frames into GIF/video output with optional recoloring.

Binary/grayscale source frames can keep their original colors, use one fixed
0/1 color pair, or use seeded random pairs that switch or ease between beats.
Both GIF and ffmpeg output consume the same per-frame color plan.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from easing import EASING_CHOICES, ease_progress
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
DEFAULT_GIF_OUTPUT_PATH = DEFAULT_OUTPUT_PATH.with_suffix(".gif")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
COLOR_TRANSITION_CHOICES = ("step", "gradient")
DEFAULT_COLOR_0 = (0, 0, 0)
DEFAULT_COLOR_1 = (255, 255, 255)
RGBColor = tuple[int, int, int]
ColorMap = tuple[RGBColor, RGBColor]
FrameColorMaps = list[ColorMap]


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


def load_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not parse metadata file: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {path}")
    return data


def metadata_easing(*metadata_sources: dict[str, object]) -> str | None:
    for metadata in metadata_sources:
        easing = metadata.get("easing")
        if isinstance(easing, str) and easing:
            return easing
    return None


def resolve_easing(
    args: argparse.Namespace,
    *metadata_sources: dict[str, object],
) -> str | None:
    """Resolve the shared frame/color easing, using linear for unlabeled gradients."""
    easing = args.easing or metadata_easing(*metadata_sources)
    if easing is None and args.color_transition == "gradient":
        return "linear"
    if args.color_transition == "gradient" and easing not in EASING_CHOICES:
        raise ValueError(
            f"Unsupported easing {easing!r} for gradient colors. "
            f"Choose one of: {', '.join(EASING_CHOICES)}."
        )
    return easing


def safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    return token.strip("-_.")


def append_token_to_name(path: Path, value: str | None) -> Path:
    """Append one normalized descriptor unless the filename already contains it."""
    if value is None:
        return path

    token = safe_filename_token(value)
    if not token or token in path.stem.lower().split("_"):
        return path

    return path.with_name(f"{path.stem}_{token}{path.suffix}")


def append_easing_to_name(path: Path, easing: str | None) -> Path:
    return append_token_to_name(path, easing)


def append_color_settings_to_name(path: Path, args: argparse.Namespace) -> Path:
    """Describe automatic colorization in a default output filename.

    Random beat colors include the resolved seed, making every unseeded render
    land at a unique path while keeping explicitly seeded renders reproducible.
    Fixed palettes include both RGB endpoint colors. Uncolored output is left
    unchanged.
    """
    if args.random_colors_per_beat:
        transition_token = (
            "-gradient" if args.color_transition == "gradient" else ""
        )
        return append_token_to_name(
            path,
            "random-colors-per-beat"
            f"{transition_token}-seed-{args.resolved_color_seed}",
        )

    if args.color_0 is None and args.color_1 is None:
        return path

    color_0 = args.color_0 or DEFAULT_COLOR_0
    color_1 = args.color_1 or DEFAULT_COLOR_1
    return append_token_to_name(
        path,
        f"colors-{format_hex_color(color_0)[1:]}-{format_hex_color(color_1)[1:]}",
    )


def default_output_name(gif: bool) -> str:
    if gif:
        return Path(DEFAULT_VIDEO_NAME).with_suffix(".gif").name
    return DEFAULT_VIDEO_NAME


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
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if not math.isfinite(bpm) or bpm <= 0:
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


# -----------------------------------------------------------------------------
# Binary color mapping
# -----------------------------------------------------------------------------


def color_channel_lookup(zero_value: int, one_value: int) -> list[int]:
    """Map every grayscale value to one channel between the 0 and 1 colors."""
    return [
        int(round(zero_value + (one_value - zero_value) * pixel / 255.0))
        for pixel in range(256)
    ]


def colorize_image(image: Image.Image, color_map: ColorMap) -> Image.Image:
    """Map grayscale 0..255 to the selected binary endpoint colors."""
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


def random_rgb_color(generator: random.Random) -> RGBColor:
    """Draw one RGB color without changing Python's process-wide random state."""
    return (
        generator.randrange(256),
        generator.randrange(256),
        generator.randrange(256),
    )


def interpolate_rgb_color(
    start: RGBColor,
    end: RGBColor,
    progress: float,
) -> RGBColor:
    """Interpolate one RGB color; progress is expected in the range 0..1."""
    return tuple(
        int(round(start[channel] * (1.0 - progress) + end[channel] * progress))
        for channel in range(3)
    )


def interpolate_color_map(
    start: ColorMap,
    end: ColorMap,
    progress: float,
) -> ColorMap:
    """Interpolate the binary 0 and 1 colors independently."""
    return (
        interpolate_rgb_color(start[0], end[0], progress),
        interpolate_rgb_color(start[1], end[1], progress),
    )


def build_random_frame_color_maps(
    frame_count: int,
    segment_frames: int,
    loop: bool,
    seed: int,
    transition: str = "step",
    easing: str = "linear",
) -> tuple[FrameColorMaps, list[ColorMap]]:
    """Create one random binary palette per beat and assign it to each frame.

    Beat anchors occur at frame indices 0, segment_frames, 2 * segment_frames,
    and so on. ``step`` holds each palette until the next beat. ``gradient``
    interpolates both the 0 and 1 colors toward the next beat using the same
    easing as the generated imagery. For a loop, the final segment gradients
    toward beat 0 and the duplicated closing frame uses beat 0 exactly.
    """
    if frame_count < 1:
        raise ValueError("At least one frame is required to create color maps.")
    if segment_frames < 1:
        raise ValueError("Frames per beat must be greater than 0.")
    if transition not in COLOR_TRANSITION_CHOICES:
        raise ValueError(f"Unsupported color transition: {transition}")
    if transition == "gradient" and easing not in EASING_CHOICES:
        raise ValueError(
            f"Unsupported easing {easing!r} for gradient colors. "
            f"Choose one of: {', '.join(EASING_CHOICES)}."
        )
    if transition == "gradient" and loop and (frame_count - 1) % segment_frames:
        raise ValueError(
            "Gradient loop sequences must contain a whole number of beats: "
            "(frame_count - 1) must be divisible by frames per beat."
        )

    generator = random.Random(seed)
    last_active_frame = (
        frame_count - 2 if loop and frame_count > 1 else frame_count - 1
    )
    beat_count = last_active_frame // segment_frames + 1

    # A partial non-loop gradient still needs the next beat palette as its
    # interpolation target, even though that target anchor is not rendered.
    if (
        transition == "gradient"
        and not loop
        and last_active_frame % segment_frames != 0
    ):
        beat_count += 1

    beat_color_maps = [
        (random_rgb_color(generator), random_rgb_color(generator))
        for _ in range(beat_count)
    ]
    frame_color_maps: FrameColorMaps = []

    for frame_index in range(frame_count):
        # A loop sequence ends with a duplicate of frame 0. Reusing its palette
        # prevents a one-frame color flash at the GIF/video loop boundary.
        if loop and frame_index > 0 and frame_index == frame_count - 1:
            frame_color_maps.append(beat_color_maps[0])
            continue

        beat_index = frame_index // segment_frames
        step = frame_index % segment_frames
        if transition == "step" or step == 0:
            frame_color_maps.append(beat_color_maps[beat_index])
            continue

        next_beat_index = (
            (beat_index + 1) % beat_count if loop else beat_index + 1
        )
        progress = ease_progress(step / segment_frames, easing)
        frame_color_maps.append(
            interpolate_color_map(
                beat_color_maps[beat_index],
                beat_color_maps[next_beat_index],
                progress,
            )
        )

    return frame_color_maps, beat_color_maps


def create_ffmpeg_sequence(
    frame_paths: list[Path],
    sequence_dir: Path,
    color_maps: FrameColorMaps | None,
) -> str:
    """Create ffmpeg's numbered input sequence, recoloring frames if requested."""
    if color_maps is not None and len(color_maps) != len(frame_paths):
        raise ValueError("Each rendered frame must have exactly one color map.")

    output_pattern = "frame_%06d.png"
    for index, frame_path in enumerate(frame_paths):
        output_path = sequence_dir / f"frame_{index:06d}.png"
        if color_maps is None:
            symlink_or_copy(frame_path, output_path)
        else:
            save_colorized_frame(frame_path, output_path, color_maps[index])
    return output_pattern


def render_video_with_ffmpeg(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    audio_path: Path | None,
    overwrite: bool,
    color_maps: FrameColorMaps | None,
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
        input_pattern = create_ffmpeg_sequence(frame_paths, temp_dir, color_maps)
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
    color_maps: FrameColorMaps | None,
) -> None:
    if color_maps is not None and len(color_maps) != len(frame_paths):
        raise ValueError("Each rendered frame must have exactly one color map.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    duration_ms = max(1, int(round(1000.0 / fps)))
    frames = [
        load_gif_frame(
            frame_path,
            None if color_maps is None else color_maps[index],
        )
        for index, frame_path in enumerate(frame_paths)
    ]
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
    color_maps: FrameColorMaps | None,
) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        if audio_path is not None:
            raise ValueError("GIF output cannot include audio. Use an MP4 output instead.")
        render_gif(frame_paths, output_path, fps, overwrite, color_maps)
        return

    if suffix in VIDEO_EXTENSIONS:
        render_video_with_ffmpeg(
            frame_paths,
            output_path,
            fps,
            audio_path,
            overwrite,
            color_maps,
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
        "--gif",
        action="store_true",
        help="Create a GIF at the automatically determined output path.",
    )
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
        "--easing",
        choices=EASING_CHOICES,
        default=None,
        help=(
            "Easing for gradient color transitions and the default output "
            "filename label. If omitted, frame/run metadata is used; an "
            "unlabeled color gradient defaults to linear."
        ),
    )
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
        "--random-colors-per-beat",
        "--random-colors",
        dest="random_colors_per_beat",
        action="store_true",
        help=(
            "Choose a new random color-pair anchor for binary values 0 and 1 "
            "at every beat. --color-transition controls whether palettes "
            "switch or interpolate between anchors."
        ),
    )
    parser.add_argument(
        "--color-seed",
        type=int,
        default=None,
        help=(
            "Seed for --random-colors-per-beat. If omitted, a random seed is "
            "chosen, included in the automatic output filename, and printed "
            "so the palette can be reproduced."
        ),
    )
    parser.add_argument(
        "--color-transition",
        "--random-color-transition",
        choices=COLOR_TRANSITION_CHOICES,
        default="step",
        help=(
            "How random beat colors change: 'step' switches on each beat; "
            "'gradient' interpolates to the next beat using the resolved "
            "--easing label."
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
        frame_metadata = load_metadata(args.frame_dir / "metadata.json")
        easing = resolve_easing(args, frame_metadata)
        args.resolved_easing = easing
        default_output = DEFAULT_GIF_OUTPUT_PATH if args.gif else DEFAULT_OUTPUT_PATH
        if args.output is None:
            args.output = append_color_settings_to_name(
                append_easing_to_name(default_output, easing),
                args,
            )
        return

    frame_subdir = validate_relative_subdir(args.frame_subdir, "--frame-subdir")
    beat_subdir = validate_relative_subdir(args.beat_subdir, "--beat-subdir")
    video_subdir = validate_relative_subdir(args.video_subdir, "--video-subdir")
    args.frame_dir = args.frame_dir or args.run_dir / frame_subdir
    args.beat_dir = args.beat_dir or args.run_dir / beat_subdir
    frame_metadata = load_metadata(args.frame_dir / "metadata.json")
    run_metadata = load_metadata(args.run_dir / "metadata.json")
    easing = resolve_easing(args, frame_metadata, run_metadata)
    args.resolved_easing = easing
    default_output = args.run_dir / video_subdir / default_output_name(args.gif)
    if args.output is None:
        args.output = append_color_settings_to_name(
            append_easing_to_name(default_output, easing),
            args,
        )


def validate_output_selection(args: argparse.Namespace) -> None:
    if args.gif and args.output.suffix.lower() != ".gif":
        raise ValueError("--gif requires a .gif output path or no --output.")
    if args.gif and args.audio is not None:
        raise ValueError("--gif cannot be used with --audio. GIF output has no audio.")


def validate_color_options(args: argparse.Namespace) -> None:
    """Reject color options whose combined meaning would be ambiguous."""
    if args.random_colors_per_beat and (
        args.color_0 is not None or args.color_1 is not None
    ):
        raise ValueError(
            "--random-colors-per-beat cannot be combined with --color-0 or --color-1."
        )
    if args.color_seed is not None and not args.random_colors_per_beat:
        raise ValueError("--color-seed requires --random-colors-per-beat.")
    if args.color_transition == "gradient" and not args.random_colors_per_beat:
        raise ValueError(
            "--color-transition gradient requires --random-colors-per-beat."
        )


def resolve_color_seed(args: argparse.Namespace) -> None:
    """Resolve random color generation to a concrete, filename-safe seed."""
    if not args.random_colors_per_beat:
        args.resolved_color_seed = None
        return

    args.resolved_color_seed = (
        args.color_seed
        if args.color_seed is not None
        else random.SystemRandom().getrandbits(64)
    )


def resolve_color_map(args: argparse.Namespace) -> ColorMap | None:
    if args.color_0 is None and args.color_1 is None:
        return None

    return (
        args.color_0 or DEFAULT_COLOR_0,
        args.color_1 or DEFAULT_COLOR_1,
    )


def resolve_frame_color_maps(
    args: argparse.Namespace,
    frame_count: int,
    segment_frames: int,
) -> tuple[FrameColorMaps | None, list[ColorMap], int | None]:
    """Resolve no coloring, one fixed palette, or a beat-indexed random plan."""
    if args.random_colors_per_beat:
        color_seed = args.resolved_color_seed
        frame_color_maps, beat_color_maps = build_random_frame_color_maps(
            frame_count=frame_count,
            segment_frames=segment_frames,
            loop=args.loop,
            seed=color_seed,
            transition=args.color_transition,
            easing=args.resolved_easing or "linear",
        )
        return frame_color_maps, beat_color_maps, color_seed

    color_map = resolve_color_map(args)
    if color_map is None:
        return None, [], None
    return [color_map] * frame_count, [color_map], None


def main() -> None:
    args = parse_args()
    validate_color_options(args)
    resolve_color_seed(args)
    resolve_run_paths(args)
    validate_output_selection(args)
    # FPS controls both output timing and duration even when beat spacing is
    # supplied directly, so it must always be a finite positive number.
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    segment_frames = (
        args.frames_per_beat
        if args.frames_per_beat is not None
        else frames_per_beat(args.fps, args.bpm)
    )
    if segment_frames <= 0:
        raise ValueError("--frames-per-beat must be greater than 0.")

    frame_paths = collect_image_paths(args.frame_dir, args.frame_pattern)
    beat_paths = (
        []
        if args.skip_beat_validation
        else collect_image_paths(args.beat_dir, args.beat_pattern)
    )

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

    color_maps, beat_color_maps, color_seed = resolve_frame_color_maps(
        args=args,
        frame_count=len(sequence_paths),
        segment_frames=segment_frames,
    )

    render_output(
        frame_paths=sequence_paths,
        output_path=args.output,
        fps=args.fps,
        audio_path=audio_path,
        overwrite=args.overwrite,
        color_maps=color_maps,
    )

    duration_seconds = len(sequence_paths) / args.fps
    if args.run_dir is not None:
        print(f"Run folder: {args.run_dir}")
    print(sequence_message)
    print(f"Frames per beat: {segment_frames}")
    print(f"Video frames: {len(sequence_paths)}")
    print(f"Frame size: {image_size[0]}x{image_size[1]}")
    if args.random_colors_per_beat:
        print(f"Random color pairs: {len(beat_color_maps)} (one per beat)")
        print(f"Color transition: {args.color_transition}")
        print(f"Color seed: {color_seed}")
    elif beat_color_maps:
        print(f"Color 0: {format_hex_color(beat_color_maps[0][0])}")
        print(f"Color 1: {format_hex_color(beat_color_maps[0][1])}")
    if args.resolved_easing is not None:
        print(f"Easing label: {args.resolved_easing}")
    print(f"Duration at {args.fps:g} fps: {duration_seconds:.2f}s")
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
