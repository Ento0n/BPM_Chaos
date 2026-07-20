from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


# The project executes modules directly from src rather than installing a
# package, so tests add the same source directory to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_diffusion_interpolation_frames import (  # noqa: E402
    DEFAULT_MODEL_ALPHA_CHECKPOINT_PATTERN,
    ModelAlphaPlan,
    build_frame_alpha_percents,
    build_frame_specs,
    build_random_binary_beat_alpha_percents,
    generate_diffusion_interpolation_frames,
    group_frame_indices_by_alpha,
    parse_args,
    resolve_model_alpha_checkpoints,
    save_metadata,
    saved_frame_alpha_percents,
    validate_model_alpha_options,
)


def metadata_args(
    run_dir: Path,
    *,
    model_alpha_mode: str,
    loop: bool,
) -> Namespace:
    """Build the generation metadata fields shared by fixed/random-mode tests."""
    return Namespace(
        run_id="test-run",
        run_parent_dir=run_dir.parent,
        run_dir=run_dir,
        frame_subdir="frames",
        beat_subdir="diffusion_beats",
        no_save_beats=False,
        run_started_at="2026-07-19T12:00:00",
        run_finished_at="2026-07-19T12:01:00",
        generation_elapsed_seconds=60.0,
        generation_elapsed="00:01:00.000",
        model_alpha_mode=model_alpha_mode,
        model_alpha_checkpoint_dir=run_dir,
        model_alpha_checkpoint_pattern="alpha_{alpha:.2f}.ckpt",
        output_dir=run_dir / "frames",
        beat_output_dir=run_dir / "diffusion_beats",
        fps=30.0,
        bpm=120.0,
        image_size=256,
        num_inference_steps=100,
        num_train_timesteps=1000,
        seed=42,
        accelerator="mps",
        require_device="mps",
        resolved_device="mps",
        scheduler="ddim",
        ddim_eta=0.0,
        interpolation="slerp",
        easing="logarithmic",
        loop=loop,
    )


class ModelAlphaScheduleTests(unittest.TestCase):
    def test_seeded_binary_beat_choices_are_local_and_reproducible(self) -> None:
        random.seed(8675309)
        global_state_before = random.getstate()

        first = build_random_binary_beat_alpha_percents(8, seed=314159)
        second = build_random_binary_beat_alpha_percents(8, seed=314159)

        self.assertEqual(first, (0, 100, 100, 0, 0, 100, 100, 0))
        self.assertEqual(first, second)
        self.assertTrue(all(percent in {0, 100} for percent in first))
        self.assertEqual(random.getstate(), global_state_before)

    def test_forward_reverse_and_unchanged_segments_are_linear(self) -> None:
        cases = [
            ((0, 100), (0, 25, 50, 75, 100)),
            ((100, 0), (100, 75, 50, 25, 0)),
            ((100, 100), (100, 100, 100, 100, 100)),
        ]

        for beat_percents, expected in cases:
            with self.subTest(beat_percents=beat_percents):
                frame_specs = build_frame_specs(2, segment_frames=4, loop=False)
                self.assertEqual(
                    build_frame_alpha_percents(frame_specs, beat_percents),
                    expected,
                )

    def test_multiple_segments_keep_exact_beat_endpoints(self) -> None:
        frame_specs = build_frame_specs(3, segment_frames=4, loop=False)

        frame_percents = build_frame_alpha_percents(
            frame_specs,
            beat_percents=(0, 100, 0),
        )

        self.assertEqual(frame_percents, (0, 25, 50, 75, 100, 75, 50, 25, 0))

    def test_non_divisor_frame_counts_round_half_up_to_nearest_percent(self) -> None:
        cases = [
            (6, (0, 17, 33, 50, 67, 83, 100)),
            (8, (0, 13, 25, 38, 50, 63, 75, 88, 100)),
        ]

        for segment_frames, expected in cases:
            with self.subTest(segment_frames=segment_frames):
                frame_specs = build_frame_specs(
                    2,
                    segment_frames=segment_frames,
                    loop=False,
                )
                self.assertEqual(
                    build_frame_alpha_percents(frame_specs, (0, 100)),
                    expected,
                )

    def test_loop_metadata_reuses_the_opening_frame_alpha(self) -> None:
        frame_specs = build_frame_specs(2, segment_frames=4, loop=True)
        frame_percents = build_frame_alpha_percents(frame_specs, (0, 100))
        plan = ModelAlphaPlan(
            seed=7,
            beat_percents=(0, 100),
            frame_percents=frame_percents,
            checkpoint_paths={},
        )

        self.assertEqual(frame_percents, (0, 25, 50, 75, 100, 75, 50, 25))
        self.assertEqual(
            saved_frame_alpha_percents(plan, loop=True),
            (0, 25, 50, 75, 100, 75, 50, 25, 0),
        )

    def test_grouping_preserves_original_timeline_indices(self) -> None:
        self.assertEqual(
            group_frame_indices_by_alpha((0, 50, 100, 50, 0)),
            {0: [0, 4], 50: [1, 3], 100: [2]},
        )


class ModelAlphaCheckpointTests(unittest.TestCase):
    def test_resolves_two_decimal_endpoint_and_intermediate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_dir = Path(temporary_dir)
            for percent in (0, 25, 100):
                path = checkpoint_dir / DEFAULT_MODEL_ALPHA_CHECKPOINT_PATTERN.format(
                    alpha=percent / 100,
                    percent=percent,
                )
                path.touch()

            resolved = resolve_model_alpha_checkpoints(
                required_percents=(100, 0, 25, 25),
                checkpoint_dir=checkpoint_dir,
                checkpoint_pattern=DEFAULT_MODEL_ALPHA_CHECKPOINT_PATTERN,
            )

            self.assertEqual(list(resolved), [0, 25, 100])
            self.assertEqual(resolved[0].name, "centered_geometric_wood_diagonals_alpha_0.00.ckpt")
            self.assertEqual(resolved[25].name, "centered_geometric_wood_diagonals_alpha_0.25.ckpt")
            self.assertEqual(
                resolved[100].name,
                "centered_geometric_wood_diagonals_alpha_1.00.ckpt",
            )

    def test_missing_required_checkpoint_reports_the_exact_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(FileNotFoundError, "alpha=0.37"):
                resolve_model_alpha_checkpoints(
                    required_percents=(37,),
                    checkpoint_dir=Path(temporary_dir),
                    checkpoint_pattern=DEFAULT_MODEL_ALPHA_CHECKPOINT_PATTERN,
                )


class ModelAlphaGenerationTests(unittest.TestCase):
    def test_random_mode_loads_each_alpha_once_and_restores_frame_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "frames"
            beat_output_dir = root / "beats"
            checkpoint_paths = {
                percent: root / f"alpha_{percent:03d}.ckpt"
                for percent in (0, 50, 100)
            }
            plan = ModelAlphaPlan(
                seed=42,
                beat_percents=(0, 100, 0),
                frame_percents=(0, 50, 100, 50, 0),
                checkpoint_paths=checkpoint_paths,
            )
            loaded_paths: list[Path] = []
            saved_paths: list[Path] = []

            class FakeModule:
                def to(self, device: torch.device) -> "FakeModule":
                    return self

            def fake_load(
                checkpoint_path: Path,
                image_size: int,
                device: torch.device,
            ) -> tuple[FakeModule, torch.device]:
                loaded_paths.append(checkpoint_path)
                return FakeModule(), device

            def fake_save(image_tensor: torch.Tensor, path: Path) -> None:
                saved_paths.append(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frame")

            fake_scheduler = SimpleNamespace(timesteps=[torch.tensor(0)])
            with (
                patch(
                    "generate_diffusion_interpolation_frames.torch.manual_seed"
                ) as manual_seed,
                patch(
                    "generate_diffusion_interpolation_frames.create_scheduler",
                    return_value=fake_scheduler,
                ),
                patch(
                    "generate_diffusion_interpolation_frames.set_scheduler_timesteps"
                ),
                patch(
                    "generate_diffusion_interpolation_frames.load_diffusion_module",
                    side_effect=fake_load,
                ),
                patch(
                    "generate_diffusion_interpolation_frames.denoise_batch",
                    side_effect=lambda **kwargs: kwargs["initial_noise"],
                ),
                patch(
                    "generate_diffusion_interpolation_frames.save_image",
                    side_effect=fake_save,
                ),
            ):
                frame_count, beat_count, device, parameter_device = (
                    generate_diffusion_interpolation_frames(
                        checkpoint_path=None,
                        model_alpha_plan=plan,
                        output_dir=output_dir,
                        beat_output_dir=beat_output_dir,
                        image_size=4,
                        num_beats=3,
                        segment_frames=2,
                        num_inference_steps=1,
                        num_train_timesteps=10,
                        seed=42,
                        accelerator="cpu",
                        scheduler_name="ddim",
                        ddim_eta=0.0,
                        interpolation="lerp",
                        easing="logarithmic",
                        batch_size=2,
                        loop=False,
                        required_device="cpu",
                        show_progress=False,
                    )
                )

            self.assertEqual(loaded_paths, [
                checkpoint_paths[0],
                checkpoint_paths[50],
                checkpoint_paths[100],
            ])
            self.assertEqual(frame_count, 5)
            self.assertEqual(beat_count, 3)
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(parameter_device, torch.device("cpu"))
            manual_seed.assert_called_once_with(42)
            self.assertEqual(
                sorted(path.name for path in saved_paths if path.parent == output_dir),
                [f"frame_{index:06d}.png" for index in range(5)],
            )
            self.assertEqual(
                sorted(path.name for path in saved_paths if path.parent == beat_output_dir),
                ["sample_000.png", "sample_001.png", "sample_002.png"],
            )

    def test_default_cli_mode_remains_fixed_and_conflicts_are_rejected(self) -> None:
        default_args = parse_args([])
        self.assertEqual(default_args.model_alpha_mode, "fixed")
        validate_model_alpha_options(default_args)

        random_with_fixed_checkpoint = parse_args(
            [
                "--model-alpha-mode",
                "random-binary",
                "--checkpoint",
                "fixed.ckpt",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--checkpoint cannot be used"):
            validate_model_alpha_options(random_with_fixed_checkpoint)


class ModelAlphaMetadataTests(unittest.TestCase):
    def test_metadata_records_complete_loop_schedule_and_checkpoint_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint_paths = {
                0: run_dir / "alpha_0.00.ckpt",
                50: run_dir / "alpha_0.50.ckpt",
                100: run_dir / "alpha_1.00.ckpt",
            }
            plan = ModelAlphaPlan(
                seed=99,
                beat_percents=(0, 100),
                frame_percents=(0, 50, 100, 50),
                checkpoint_paths=checkpoint_paths,
            )
            args = metadata_args(
                run_dir,
                model_alpha_mode="random-binary",
                loop=True,
            )

            save_metadata(
                checkpoint_path=None,
                model_alpha_plan=plan,
                args=args,
                segment_frames=2,
                frame_count=5,
                beat_count=2,
            )
            metadata = json.loads((run_dir / "metadata.json").read_text())

            self.assertIsNone(metadata["checkpoint"])
            self.assertEqual(metadata["model_alpha_mode"], "random-binary")
            self.assertEqual(metadata["model_alpha_seed"], 99)
            self.assertEqual(metadata["model_alpha_beat_values"], [0.0, 1.0])
            self.assertEqual(
                metadata["model_alpha_frame_values"],
                [0.0, 0.5, 1.0, 0.5, 0.0],
            )
            self.assertEqual(
                len(metadata["model_alpha_frame_values"]),
                metadata["frame_count"],
            )
            self.assertEqual(
                sorted(metadata["model_alpha_checkpoints"]),
                ["0.00", "0.50", "1.00"],
            )
            self.assertFalse(metadata["model_alpha_uses_noise_easing"])

    def test_fixed_metadata_keeps_the_legacy_checkpoint_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint_path = run_dir / "fixed.ckpt"
            args = metadata_args(
                run_dir,
                model_alpha_mode="fixed",
                loop=False,
            )

            save_metadata(
                checkpoint_path=checkpoint_path,
                model_alpha_plan=None,
                args=args,
                segment_frames=1,
                frame_count=2,
                beat_count=2,
            )
            metadata = json.loads((run_dir / "metadata.json").read_text())

            self.assertEqual(metadata["checkpoint"], str(checkpoint_path))
            self.assertEqual(metadata["model_alpha_mode"], "fixed")
            self.assertIsNone(metadata["model_alpha_seed"])
            self.assertIsNone(metadata["model_alpha_frame_values"])
            self.assertIsNone(metadata["model_alpha_checkpoints"])


if __name__ == "__main__":
    unittest.main()
