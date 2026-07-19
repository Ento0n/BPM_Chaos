"""Create one Lightning diffusion checkpoint by interpolating two checkpoints.

The command loads both checkpoints on CPU, validates that every state tensor has
the same name, shape, and dtype, and computes this parameter-wise interpolation:

    mixed = (1 - alpha) * checkpoint_a + alpha * checkpoint_b

The output keeps the model-construction metadata from checkpoint A, adds source
provenance, and intentionally omits optimizer and Trainer state. It can therefore
be used for inference or loaded as the starting model for fresh fine-tuning, but
it is not a resumable snapshot of either source training run.
"""

from __future__ import annotations

import argparse
import copy
import math
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch


# =============================================================================
# Checkpoint loading, validation, and interpolation
# =============================================================================


def load_lightning_checkpoint(path: Path) -> dict:
    """Load a trusted Lightning checkpoint on CPU and validate its core fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    # The merge only needs tensors and primitive Lightning metadata. Restricted
    # loading avoids executing arbitrary pickle payloads from checkpoint files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"Checkpoint has no non-empty state_dict: {path}")
    if "pytorch-lightning_version" not in checkpoint:
        raise ValueError(f"Checkpoint has no pytorch-lightning version: {path}")
    if "hyper_parameters" not in checkpoint:
        raise ValueError(f"Checkpoint has no model hyperparameters: {path}")

    return checkpoint


def merge_checkpoint_files(
    checkpoint_a_path: Path,
    checkpoint_b_path: Path,
    alpha: float,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int | bool]:
    """Interpolate two compatible Lightning checkpoints and save the result."""
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"Alpha must be a finite value between 0 and 1; got {alpha!r}"
        )

    checkpoint_a_path = checkpoint_a_path.expanduser().resolve()
    checkpoint_b_path = checkpoint_b_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    # Never allow an output checkpoint to replace either input, even when the
    # caller explicitly allows overwriting an existing output file.
    if output_path in {checkpoint_a_path, checkpoint_b_path}:
        raise ValueError("The output path must be different from both input checkpoints")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    checkpoint_a = load_lightning_checkpoint(checkpoint_a_path)
    checkpoint_b = load_lightning_checkpoint(checkpoint_b_path)
    state_a = checkpoint_a["state_dict"]
    state_b = checkpoint_b["state_dict"]

    # Matching names are essential: identical tensor shapes alone do not prove
    # that parameters occupy the same semantic positions in the architecture.
    keys_a = list(state_a)
    keys_b = list(state_b)
    if set(keys_a) != set(keys_b):
        missing_from_b = [name for name in keys_a if name not in state_b]
        extra_in_b = [name for name in keys_b if name not in state_a]
        raise ValueError(
            "Checkpoint state_dict keys differ. "
            f"Missing from B: {missing_from_b[:5] or 'none'}; "
            f"extra in B: {extra_in_b[:5] or 'none'}"
        )

    merged_state: dict[str, torch.Tensor] = {}
    non_floating_tensor_count = 0
    for name in keys_a:
        tensor_a = state_a[name]
        tensor_b = state_b[name]
        if not torch.is_tensor(tensor_a) or not torch.is_tensor(tensor_b):
            raise ValueError(
                f"state_dict entry {name!r} is not a tensor in both checkpoints"
            )
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Shape mismatch for {name!r}: {tuple(tensor_a.shape)} != "
                f"{tuple(tensor_b.shape)}"
            )
        if tensor_a.dtype != tensor_b.dtype:
            raise ValueError(
                f"Dtype mismatch for {name!r}: {tensor_a.dtype} != {tensor_b.dtype}"
            )

        # Floating-point and complex parameters support a meaningful weighted
        # midpoint. Integer and boolean buffers cannot be interpolated without
        # inventing rounding semantics, so use the nearer endpoint instead.
        if tensor_a.is_floating_point() or tensor_a.is_complex():
            # Clone at the endpoints so alpha 0 and 1 are bit-exact copies and
            # do not perform arithmetic involving the unused source tensor.
            if alpha == 0.0:
                merged_tensor = tensor_a.clone()
            elif alpha == 1.0:
                merged_tensor = tensor_b.clone()
            else:
                merged_tensor = torch.lerp(tensor_a, tensor_b, alpha)
            if not torch.isfinite(merged_tensor).all().item():
                raise ValueError(f"Interpolation produced non-finite values for {name!r}")
        else:
            non_floating_tensor_count += 1
            merged_tensor = (tensor_a if alpha <= 0.5 else tensor_b).clone()
        merged_state[name] = merged_tensor

    hyper_parameters_match = (
        checkpoint_a["hyper_parameters"] == checkpoint_b["hyper_parameters"]
    )

    # Keep only what Lightning needs to construct and load the model. Optimizer,
    # callback, scheduler, and loop state from either source would be stale and
    # misleading for the newly interpolated parameters.
    merged_checkpoint = {
        "pytorch-lightning_version": checkpoint_a["pytorch-lightning_version"],
        "state_dict": merged_state,
        "hparams_name": checkpoint_a.get("hparams_name", "kwargs"),
        "hyper_parameters": copy.deepcopy(checkpoint_a["hyper_parameters"]),
        "merge_metadata": {
            "method": "linear_parameter_interpolation",
            "formula": "(1 - alpha) * checkpoint_a + alpha * checkpoint_b",
            "alpha": alpha,
            "checkpoint_a": str(checkpoint_a_path),
            "checkpoint_b": str(checkpoint_b_path),
            "checkpoint_a_epoch": checkpoint_a.get("epoch"),
            "checkpoint_b_epoch": checkpoint_b.get("epoch"),
            "checkpoint_a_global_step": checkpoint_a.get("global_step"),
            "checkpoint_b_global_step": checkpoint_b.get("global_step"),
            "hyper_parameters_source": "checkpoint_a",
            "source_hyper_parameters_match": hyper_parameters_match,
            "non_floating_tensor_policy": (
                "checkpoint_a when alpha <= 0.5, otherwise checkpoint_b"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "intended_use": "inference_or_fresh_optimizer_finetuning",
        },
    }

    # Write beside the destination and rename only after serialization succeeds,
    # preventing a failed or interrupted save from leaving a partial checkpoint.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.save(merged_checkpoint, temporary_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output was created while merging and will not be replaced: {output_path}"
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "tensor_count": len(merged_state),
        "parameter_count": sum(tensor.numel() for tensor in merged_state.values()),
        "non_floating_tensor_count": non_floating_tensor_count,
        "hyper_parameters_match": hyper_parameters_match,
    }


# =============================================================================
# Command-line interface
# =============================================================================


def alpha_argument(value: str) -> float:
    """Parse an interpolation alpha and provide an argparse-friendly error."""
    try:
        alpha = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "alpha must be a number between 0 and 1"
        ) from error
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be a finite number between 0 and 1")
    return alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate two compatible Lightning diffusion checkpoints.",
        epilog=(
            "alpha=0 selects checkpoint A, alpha=0.5 is an equal mix, "
            "and alpha=1 selects checkpoint B."
        ),
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--alpha", type=alpha_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing output file. Input checkpoints are always protected."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = merge_checkpoint_files(
            checkpoint_a_path=args.checkpoint_a,
            checkpoint_b_path=args.checkpoint_b,
            alpha=args.alpha,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    output_path = args.output.expanduser().resolve()
    print(f"Saved mixed checkpoint: {output_path}")
    print(
        f"Alpha: {args.alpha} "
        f"(A weight={1.0 - args.alpha}, B weight={args.alpha})"
    )
    print(
        "State dict: "
        f"{summary['tensor_count']} tensors, "
        f"{summary['parameter_count']:,} values, "
        f"{summary['non_floating_tensor_count']} non-floating tensors"
    )
    if not summary["hyper_parameters_match"]:
        print(
            "Warning: source hyperparameters differ; "
            "the output uses checkpoint A's values."
        )
    print(f"Output size: {output_path.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
