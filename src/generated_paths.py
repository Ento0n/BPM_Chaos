from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = PROJECT_ROOT / "generated"
DEFAULT_DIFFUSION_BEAT_SUBDIR = "diffusion_beats"
DEFAULT_FRAME_SUBDIR = "frames"
DEFAULT_VIDEO_SUBDIR = "videos"
DEFAULT_VIDEO_NAME = "preview.mp4"


def timestamp_run_id() -> str:
    return datetime.now().strftime("%y_%m_%d-%H_%M_%S")


def validate_relative_subdir(value: str, argument_name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{argument_name} must be a relative subdirectory name.")
    return path


def create_unique_run_dir(parent_dir: Path) -> tuple[str, Path]:
    parent_dir.mkdir(parents=True, exist_ok=True)

    while True:
        run_id = timestamp_run_id()
        run_dir = parent_dir / run_id
        if run_dir.exists():
            time.sleep(1.0)
            continue

        run_dir.mkdir(parents=True, exist_ok=False)
        return run_id, run_dir
