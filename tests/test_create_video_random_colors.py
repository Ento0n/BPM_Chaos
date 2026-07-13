from __future__ import annotations

import random
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image


# The project keeps executable modules directly in src/ rather than in an
# installed package. Add that directory exactly as command-line execution does
# so these tests can run from either the repository root or the tests folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from create_video_from_frames import (  # noqa: E402
    DEFAULT_COLOR_0,
    DEFAULT_COLOR_1,
    append_color_settings_to_name,
    build_random_frame_color_maps,
    colorize_image,
    interpolate_color_map,
    resolve_color_seed,
    resolve_frame_color_maps,
    resolve_run_paths,
    validate_color_options,
)
from easing import ease_progress  # noqa: E402


def color_args(
    *,
    color_0: tuple[int, int, int] | None = None,
    color_1: tuple[int, int, int] | None = None,
    random_colors_per_beat: bool = False,
    color_seed: int | None = None,
    color_transition: str = "step",
    loop: bool = False,
) -> Namespace:
    """Build the subset of parsed command-line arguments used by color logic."""
    return Namespace(
        color_0=color_0,
        color_1=color_1,
        random_colors_per_beat=random_colors_per_beat,
        color_seed=color_seed,
        resolved_color_seed=color_seed,
        color_transition=color_transition,
        resolved_easing="linear",
        loop=loop,
    )


class RandomBeatPaletteTests(unittest.TestCase):
    def test_seeded_beat_palettes_are_deterministic(self) -> None:
        expected_beat_palettes = [
            ((98, 140, 146), (50, 60, 156)),
            ((158, 94, 86), (17, 105, 79)),
            ((95, 163, 1), (69, 133, 171)),
        ]

        first_frames, first_beats = build_random_frame_color_maps(
            frame_count=6,
            segment_frames=2,
            loop=False,
            seed=314159,
        )
        second_frames, second_beats = build_random_frame_color_maps(
            frame_count=6,
            segment_frames=2,
            loop=False,
            seed=314159,
        )

        self.assertEqual(first_beats, expected_beat_palettes)
        self.assertEqual(first_frames, second_frames)
        self.assertEqual(first_beats, second_beats)

    def test_non_loop_palette_changes_only_at_beat_boundaries(self) -> None:
        # Seven frames at three frames per beat have anchors at 0, 3, and 6.
        # The last frame is a real non-loop endpoint, so it receives a new
        # third palette instead of reusing the opening palette.
        frame_palettes, beat_palettes = build_random_frame_color_maps(
            frame_count=7,
            segment_frames=3,
            loop=False,
            seed=23,
        )

        self.assertEqual(len(beat_palettes), 3)
        self.assertEqual(frame_palettes[0:3], [beat_palettes[0]] * 3)
        self.assertEqual(frame_palettes[3:6], [beat_palettes[1]] * 3)
        self.assertEqual(frame_palettes[6], beat_palettes[2])
        self.assertNotEqual(frame_palettes[6], frame_palettes[0])

    def test_loop_closing_frame_reuses_opening_palette(self) -> None:
        # In a closed sequence frame 6 duplicates frame 0. It must reuse beat
        # zero's palette and must not create an unused third random palette.
        frame_palettes, beat_palettes = build_random_frame_color_maps(
            frame_count=7,
            segment_frames=3,
            loop=True,
            seed=23,
        )

        self.assertEqual(len(beat_palettes), 2)
        self.assertEqual(frame_palettes[0:3], [beat_palettes[0]] * 3)
        self.assertEqual(frame_palettes[3:6], [beat_palettes[1]] * 3)
        self.assertEqual(frame_palettes[-1], frame_palettes[0])

    def test_single_frame_loop_has_one_valid_palette(self) -> None:
        # A one-frame loop is both its opening and closing frame. The planner
        # still needs to generate beat zero before any closing-frame reuse.
        frame_palettes, beat_palettes = build_random_frame_color_maps(
            frame_count=1,
            segment_frames=4,
            loop=True,
            seed=5,
        )

        self.assertEqual(len(beat_palettes), 1)
        self.assertEqual(frame_palettes, beat_palettes)

    def test_palette_generation_does_not_change_global_random_state(self) -> None:
        random.seed(8675309)
        state_before = random.getstate()

        build_random_frame_color_maps(
            frame_count=9,
            segment_frames=2,
            loop=False,
            seed=101,
        )

        self.assertEqual(random.getstate(), state_before)

    def test_gradient_uses_the_selected_easing_between_beat_palettes(self) -> None:
        for easing in ("linear", "cosine", "logarithmic"):
            with self.subTest(easing=easing):
                frame_palettes, beat_palettes = build_random_frame_color_maps(
                    frame_count=9,
                    segment_frames=4,
                    loop=False,
                    seed=314159,
                    transition="gradient",
                    easing=easing,
                )

                # Beat frames stay exact. Between them, both binary endpoint
                # colors follow the selected image-frame easing function.
                self.assertEqual(frame_palettes[0], beat_palettes[0])
                self.assertEqual(frame_palettes[4], beat_palettes[1])
                self.assertEqual(frame_palettes[8], beat_palettes[2])
                for step in range(1, 4):
                    expected = interpolate_color_map(
                        beat_palettes[0],
                        beat_palettes[1],
                        ease_progress(step / 4, easing),
                    )
                    self.assertEqual(frame_palettes[step], expected)

    def test_one_frame_per_beat_keeps_gradient_anchors_exact(self) -> None:
        frame_palettes, beat_palettes = build_random_frame_color_maps(
            frame_count=4,
            segment_frames=1,
            loop=False,
            seed=23,
            transition="gradient",
            easing="logarithmic",
        )

        self.assertEqual(frame_palettes, beat_palettes)

    def test_loop_gradient_returns_smoothly_to_opening_palette(self) -> None:
        frame_palettes, beat_palettes = build_random_frame_color_maps(
            frame_count=9,
            segment_frames=4,
            loop=True,
            seed=23,
            transition="gradient",
            easing="linear",
        )

        self.assertEqual(len(beat_palettes), 2)
        self.assertEqual(frame_palettes[4], beat_palettes[1])
        for step in range(1, 4):
            expected = interpolate_color_map(
                beat_palettes[1],
                beat_palettes[0],
                step / 4,
            )
            self.assertEqual(frame_palettes[4 + step], expected)
        self.assertEqual(frame_palettes[-1], beat_palettes[0])

    def test_partial_gradient_has_a_future_target_palette(self) -> None:
        frame_palettes, beat_palettes = build_random_frame_color_maps(
            frame_count=3,
            segment_frames=4,
            loop=False,
            seed=23,
            transition="gradient",
            easing="linear",
        )

        self.assertEqual(len(beat_palettes), 2)
        self.assertEqual(
            frame_palettes[2],
            interpolate_color_map(beat_palettes[0], beat_palettes[1], 0.5),
        )

    def test_partial_gradient_loop_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Gradient loop sequences must contain a whole number of beats",
        ):
            build_random_frame_color_maps(
                frame_count=8,
                segment_frames=4,
                loop=True,
                seed=23,
                transition="gradient",
                easing="linear",
            )

    def test_gradient_rejects_an_unknown_easing(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported easing 'unknown'"):
            build_random_frame_color_maps(
                frame_count=1,
                segment_frames=1,
                loop=False,
                seed=23,
                transition="gradient",
                easing="unknown",
            )


class BinaryColorMappingTests(unittest.TestCase):
    def test_binary_values_map_exactly_to_selected_endpoint_colors(self) -> None:
        source = Image.new("L", (2, 1))
        source.putdata([0, 255])
        color_0 = (12, 34, 56)
        color_1 = (210, 190, 170)

        colorized = colorize_image(source, (color_0, color_1))
        try:
            self.assertEqual(colorized.getpixel((0, 0)), color_0)
            self.assertEqual(colorized.getpixel((1, 0)), color_1)
        finally:
            colorized.close()
            source.close()


class ColorOptionTests(unittest.TestCase):
    def test_fixed_colors_fill_missing_binary_endpoint_defaults(self) -> None:
        cases = [
            (
                color_args(color_0=(11, 22, 33)),
                ((11, 22, 33), DEFAULT_COLOR_1),
            ),
            (
                color_args(color_1=(44, 55, 66)),
                (DEFAULT_COLOR_0, (44, 55, 66)),
            ),
        ]

        for args, expected_palette in cases:
            with self.subTest(expected_palette=expected_palette):
                frame_palettes, beat_palettes, seed = resolve_frame_color_maps(
                    args,
                    frame_count=3,
                    segment_frames=2,
                )

                self.assertEqual(frame_palettes, [expected_palette] * 3)
                self.assertEqual(beat_palettes, [expected_palette])
                self.assertIsNone(seed)

    def test_random_colors_conflict_with_either_fixed_color(self) -> None:
        for fixed_colors in (
            {"color_0": (1, 2, 3)},
            {"color_1": (4, 5, 6)},
            {"color_0": (1, 2, 3), "color_1": (4, 5, 6)},
        ):
            with self.subTest(fixed_colors=fixed_colors):
                args = color_args(random_colors_per_beat=True, **fixed_colors)
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot be combined with --color-0 or --color-1",
                ):
                    validate_color_options(args)

    def test_color_seed_requires_random_colors(self) -> None:
        args = color_args(color_seed=1234)

        with self.assertRaisesRegex(
            ValueError,
            "--color-seed requires --random-colors-per-beat",
        ):
            validate_color_options(args)

    def test_gradient_transition_requires_random_beat_colors(self) -> None:
        args = color_args(color_transition="gradient")

        with self.assertRaisesRegex(
            ValueError,
            "--color-transition gradient requires --random-colors-per-beat",
        ):
            validate_color_options(args)


class ColorOutputFilenameTests(unittest.TestCase):
    def test_color_settings_are_included_in_automatic_names(self) -> None:
        cases = [
            (color_args(), "preview.mp4"),
            (
                color_args(color_0=(17, 34, 51)),
                "preview_colors-112233-ffffff.mp4",
            ),
            (
                color_args(color_1=(170, 187, 204)),
                "preview_colors-000000-aabbcc.mp4",
            ),
            (
                color_args(
                    random_colors_per_beat=True,
                    color_seed=42,
                ),
                "preview_random-colors-per-beat-seed-42.mp4",
            ),
            (
                color_args(
                    random_colors_per_beat=True,
                    color_seed=42,
                    color_transition="gradient",
                ),
                "preview_random-colors-per-beat-gradient-seed-42.mp4",
            ),
        ]

        for args, expected_name in cases:
            with self.subTest(expected_name=expected_name):
                output = append_color_settings_to_name(Path("preview.mp4"), args)
                self.assertEqual(output.name, expected_name)

    def test_generated_seed_is_reused_by_filename_and_palette(self) -> None:
        args = color_args(
            random_colors_per_beat=True,
            color_transition="gradient",
        )
        with patch(
            "create_video_from_frames.random.SystemRandom"
        ) as system_random:
            system_random.return_value.getrandbits.return_value = 987654321
            resolve_color_seed(args)

        output = append_color_settings_to_name(Path("preview.gif"), args)
        _, _, palette_seed = resolve_frame_color_maps(
            args,
            frame_count=3,
            segment_frames=2,
        )

        self.assertEqual(args.resolved_color_seed, 987654321)
        self.assertEqual(palette_seed, 987654321)
        self.assertEqual(
            output.name,
            "preview_random-colors-per-beat-gradient-seed-987654321.gif",
        )

    def test_default_name_combines_easing_and_color_settings(self) -> None:
        args = color_args(
            color_0=(17, 34, 51),
            color_1=(170, 187, 204),
        )
        args.run_dir = Path("generated/example")
        args.frame_subdir = "frames"
        args.beat_subdir = "diffusion_beats"
        args.video_subdir = "videos"
        args.frame_dir = None
        args.beat_dir = None
        args.output = None
        args.gif = False
        args.easing = "cosine"

        resolve_run_paths(args)

        self.assertEqual(
            args.output,
            Path(
                "generated/example/videos/"
                "preview_cosine_colors-112233-aabbcc.mp4"
            ),
        )

    def test_default_gradient_name_uses_the_resolved_easing(self) -> None:
        args = color_args(
            random_colors_per_beat=True,
            color_seed=42,
            color_transition="gradient",
        )
        args.run_dir = Path("generated/example")
        args.frame_subdir = "frames"
        args.beat_subdir = "diffusion_beats"
        args.video_subdir = "videos"
        args.frame_dir = None
        args.beat_dir = None
        args.output = None
        args.gif = True
        args.easing = "cosine"

        resolve_run_paths(args)

        self.assertEqual(args.resolved_easing, "cosine")
        self.assertEqual(
            args.output,
            Path(
                "generated/example/videos/"
                "preview_cosine_random-colors-per-beat-gradient-seed-42.gif"
            ),
        )

    def test_unlabeled_gradient_defaults_to_linear_easing(self) -> None:
        args = color_args(
            random_colors_per_beat=True,
            color_seed=42,
            color_transition="gradient",
        )
        args.run_dir = Path("generated/no-metadata")
        args.frame_subdir = "frames"
        args.beat_subdir = "diffusion_beats"
        args.video_subdir = "videos"
        args.frame_dir = None
        args.beat_dir = None
        args.output = None
        args.gif = False
        args.easing = None

        resolve_run_paths(args)

        self.assertEqual(args.resolved_easing, "linear")
        self.assertEqual(
            args.output.name,
            "preview_linear_random-colors-per-beat-gradient-seed-42.mp4",
        )

    def test_explicit_output_name_is_preserved(self) -> None:
        args = color_args(
            random_colors_per_beat=True,
            color_seed=42,
            color_transition="gradient",
        )
        args.run_dir = Path("generated/example")
        args.frame_subdir = "frames"
        args.beat_subdir = "diffusion_beats"
        args.video_subdir = "videos"
        args.frame_dir = None
        args.beat_dir = None
        args.output = Path("custom/exact-name.mp4")
        args.gif = False
        args.easing = "cosine"

        resolve_run_paths(args)

        self.assertEqual(args.output, Path("custom/exact-name.mp4"))


if __name__ == "__main__":
    unittest.main()
