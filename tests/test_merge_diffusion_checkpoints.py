from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import torch


# The merge command is an executable module in src rather than an installed
# package. Add src exactly as command-line execution does for direct unit tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merge_diffusion_checkpoints import (  # noqa: E402
    alpha_argument,
    merge_checkpoint_files,
)


def save_tiny_checkpoint(
    path: Path,
    *,
    weight: torch.Tensor,
    counter: torch.Tensor | None = None,
) -> None:
    """Save the minimum Lightning checkpoint needed by the merge command."""
    state_dict = {"model.weight": weight}
    if counter is not None:
        state_dict["model.counter"] = counter
    torch.save(
        {
            "epoch": 3,
            "global_step": 120,
            "pytorch-lightning_version": "2.6.5",
            "state_dict": state_dict,
            "optimizer_states": [{"stale": True}],
            "callbacks": {"stale": True},
            "hparams_name": "kwargs",
            "hyper_parameters": {"image_size": 16, "learning_rate": 1e-4},
        },
        path,
    )


class MergeCheckpointTests(unittest.TestCase):
    def test_writes_exact_interpolation_and_lightweight_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            checkpoint_a = directory / "a.ckpt"
            checkpoint_b = directory / "b.ckpt"
            output = directory / "nested" / "mixed.ckpt"
            save_tiny_checkpoint(
                checkpoint_a,
                weight=torch.tensor([0.0, 4.0]),
                counter=torch.tensor(10, dtype=torch.int64),
            )
            save_tiny_checkpoint(
                checkpoint_b,
                weight=torch.tensor([8.0, 12.0]),
                counter=torch.tensor(20, dtype=torch.int64),
            )

            summary = merge_checkpoint_files(
                checkpoint_a,
                checkpoint_b,
                alpha=0.25,
                output_path=output,
            )
            result = torch.load(output, map_location="cpu", weights_only=True)

            torch.testing.assert_close(
                result["state_dict"]["model.weight"],
                torch.tensor([2.0, 6.0]),
            )
            # Integer buffers use the nearer endpoint instead of lossy rounding.
            self.assertEqual(result["state_dict"]["model.counter"].item(), 10)
            self.assertEqual(result["merge_metadata"]["alpha"], 0.25)
            self.assertEqual(
                result["merge_metadata"]["checkpoint_a"],
                str(checkpoint_a.resolve()),
            )
            self.assertNotIn("optimizer_states", result)
            self.assertNotIn("callbacks", result)
            self.assertEqual(summary["tensor_count"], 2)
            self.assertEqual(summary["parameter_count"], 3)
            self.assertEqual(summary["non_floating_tensor_count"], 1)

    def test_alpha_endpoints_are_bit_exact_copies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            checkpoint_a = directory / "a.ckpt"
            checkpoint_b = directory / "b.ckpt"
            tensor_a = torch.tensor([1.234567, -9.876543], dtype=torch.float32)
            tensor_b = torch.tensor([-4.5, 6.75], dtype=torch.float32)
            save_tiny_checkpoint(checkpoint_a, weight=tensor_a)
            save_tiny_checkpoint(checkpoint_b, weight=tensor_b)

            for alpha, expected in ((0.0, tensor_a), (1.0, tensor_b)):
                with self.subTest(alpha=alpha):
                    output = directory / f"mixed-{alpha}.ckpt"
                    merge_checkpoint_files(
                        checkpoint_a,
                        checkpoint_b,
                        alpha=alpha,
                        output_path=output,
                    )
                    result = torch.load(output, map_location="cpu", weights_only=True)
                    self.assertTrue(
                        torch.equal(result["state_dict"]["model.weight"], expected)
                    )

    def test_rejects_mismatched_state_dicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            checkpoint_a = directory / "a.ckpt"
            checkpoint_b = directory / "b.ckpt"
            save_tiny_checkpoint(checkpoint_a, weight=torch.zeros(2))
            save_tiny_checkpoint(checkpoint_b, weight=torch.zeros(3))

            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                merge_checkpoint_files(
                    checkpoint_a,
                    checkpoint_b,
                    alpha=0.5,
                    output_path=directory / "mixed.ckpt",
                )

    def test_refuses_to_replace_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            checkpoint_a = directory / "a.ckpt"
            checkpoint_b = directory / "b.ckpt"
            output = directory / "mixed.ckpt"
            save_tiny_checkpoint(checkpoint_a, weight=torch.zeros(1))
            save_tiny_checkpoint(checkpoint_b, weight=torch.ones(1))
            output.write_bytes(b"keep me")

            with self.assertRaisesRegex(FileExistsError, "Pass --overwrite"):
                merge_checkpoint_files(
                    checkpoint_a,
                    checkpoint_b,
                    alpha=0.5,
                    output_path=output,
                )
            self.assertEqual(output.read_bytes(), b"keep me")

    def test_alpha_parser_rejects_values_outside_interpolation_range(self) -> None:
        for invalid_alpha in ("-0.1", "1.1", "nan", "infinity", "not-a-number"):
            with self.subTest(alpha=invalid_alpha):
                with self.assertRaises(argparse.ArgumentTypeError):
                    alpha_argument(invalid_alpha)


if __name__ == "__main__":
    unittest.main()
