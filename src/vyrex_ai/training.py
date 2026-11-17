"""Training utilities for VYREX AI."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch random-number-generator states."""
    state = {
        "torch_rng_state": torch.get_rng_state().cpu(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = [
            item.cpu() for item in torch.cuda.get_rng_state_all()
        ]
    return state


def _validate_rng_state(state: dict[str, Any]) -> None:
    """Validate all supplied RNG states without changing global RNG state."""
    if not isinstance(state, dict):
        raise TypeError("Checkpoint RNG state must be a dictionary")

    required_cpu_states = {
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
    }
    if state:
        missing = required_cpu_states - state.keys()
        if missing:
            raise ValueError(
                "Checkpoint RNG state is incomplete; missing required RNG "
                f"states: {sorted(missing)}"
            )

    if "torch_rng_state" in state:
        value = state["torch_rng_state"]
        if not torch.is_tensor(value):
            raise TypeError("Checkpoint PyTorch RNG state must be a tensor")
        try:
            torch.Generator(device="cpu").set_state(value.detach().cpu())
        except Exception as exc:
            raise ValueError(
                "Checkpoint PyTorch RNG state is invalid"
            ) from exc

    if "numpy_rng_state" in state:
        try:
            np.random.RandomState().set_state(state["numpy_rng_state"])
        except Exception as exc:
            raise ValueError(
                "Checkpoint NumPy RNG state is invalid"
            ) from exc

    if "python_rng_state" in state:
        try:
            random.Random().setstate(state["python_rng_state"])
        except Exception as exc:
            raise ValueError(
                "Checkpoint Python RNG state is invalid"
            ) from exc

    if "cuda_rng_state_all" in state:
        saved = state["cuda_rng_state_all"]
        if not isinstance(saved, (list, tuple)):
            raise TypeError(
                "Checkpoint CUDA RNG state must be a list or tuple"
            )
        if any(not torch.is_tensor(item) for item in saved):
            raise TypeError(
                "Checkpoint CUDA RNG states must contain only tensors"
            )

        if torch.cuda.is_available():
            if len(saved) != torch.cuda.device_count():
                raise ValueError(
                    "Checkpoint CUDA RNG state count does not match "
                    "the available CUDA devices"
                )
            for index, item in enumerate(saved):
                try:
                    torch.Generator(
                        device=f"cuda:{index}"
                    ).set_state(item.cpu())
                except Exception as exc:
                    raise ValueError(
                        "Checkpoint CUDA RNG state for device "
                        f"{index} is invalid"
                    ) from exc


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore validated Python, NumPy, and PyTorch RNG states."""
    _validate_rng_state(state)

    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all([
            item.cpu() for item in state["cuda_rng_state_all"]
        ])


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: nn.Module,
    training_config: Any,
) -> torch.optim.Optimizer:
    """Create the AdamW optimizer from the project configuration."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=tuple(training_config.betas),
        weight_decay=training_config.weight_decay,
    )


def learning_rate_at_step(step: int, training_config: Any) -> float:
    """Calculate warmup followed by cosine decay."""
    warmup_steps = training_config.warmup_steps
    max_steps = training_config.max_steps
    peak_lr = training_config.learning_rate
    minimum_lr = training_config.min_learning_rate

    if step < 0:
        raise ValueError("step cannot be negative")

    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(
        1.0,
        max(0.0, (step - warmup_steps) / decay_steps),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_lr + cosine * (peak_lr - minimum_lr)


def _validate_resume_signature(
    signature: dict[str, Any],
    name: str,
) -> None:
    """Validate the exact fields used to establish resume compatibility."""
    import string

    required = {
        "train_data_sha256",
        "validation_data_sha256",
        "tokenizer_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "sequence_length",
        "micro_batch_size",
        "gradient_accumulation_steps",
    }

    if not isinstance(signature, dict):
        raise TypeError(f"{name} must be a dictionary")

    if signature.keys() != required:
        raise ValueError(
            f"{name} must contain exactly these fields: "
            f"{', '.join(sorted(required))}"
        )

    digest_fields = {
        "train_data_sha256",
        "validation_data_sha256",
        "tokenizer_sha256",
        "model_config_sha256",
        "training_config_sha256",
    }
    for field in digest_fields:
        value = signature[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in string.hexdigits for character in value)
        ):
            raise ValueError(
                f"{name}.{field} must be a SHA-256 hex digest"
            )

    integer_fields = {
        "sequence_length",
        "micro_batch_size",
        "gradient_accumulation_steps",
    }
    for field in integer_fields:
        value = signature[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name}.{field} must be a positive integer"
            )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_validation_loss: float | None = None,
    scaler: Any | None = None,
    batches_in_current_pass: int = 0,
    resume_signature: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint atomically to avoid incomplete checkpoint files."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("Checkpoint step must be a non-negative integer")
    if step < 0:
        raise ValueError("Checkpoint step cannot be negative")

    if best_validation_loss is not None and (
        isinstance(best_validation_loss, bool)
        or not isinstance(best_validation_loss, (int, float))
        or math.isnan(best_validation_loss)
        or best_validation_loss == -math.inf
    ):
        raise ValueError(
            "Checkpoint best_validation_loss must be a real number "
            "other than NaN or negative infinity, or None"
        )

    if (
        isinstance(batches_in_current_pass, bool)
        or not isinstance(batches_in_current_pass, int)
        or batches_in_current_pass < 0
    ):
        raise ValueError(
            "Checkpoint batches_in_current_pass must be a "
            "non-negative integer"
        )

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if resume_signature is not None:
        _validate_resume_signature(resume_signature, "resume_signature")

    checkpoint = {
        "step": step,
        "batches_in_current_pass": batches_in_current_pass,
        "resume_signature": resume_signature,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_validation_loss,
        **capture_rng_state(),
    }

    if scaler is not None and scaler.is_enabled():
        checkpoint["grad_scaler_state_dict"] = scaler.state_dict()

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = [
            item.cpu() for item in torch.cuda.get_rng_state_all()
        ]

    import os
    import tempfile

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        os.close(fd)
        torch.save(checkpoint, temporary_path)

        with temporary_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _validate_checkpoint_restore_state(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
) -> None:
    """Validate all restorable state before mutating live training objects."""
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise TypeError(
            "Checkpoint model_state_dict must be a dictionary"
        )

    current_state = model.state_dict()
    if model_state.keys() != current_state.keys():
        raise ValueError(
            "Checkpoint model_state_dict keys do not match the current model"
        )

    for key, current_value in current_state.items():
        if not torch.is_tensor(current_value):
            continue

        saved_value = model_state[key]
        if not torch.is_tensor(saved_value):
            raise ValueError(
                f"Checkpoint model state entry {key!r} must be a tensor"
            )
        if saved_value.shape != current_value.shape:
            raise ValueError(
                f"Checkpoint model state shape mismatch for {key!r}"
            )

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError(
                "Checkpoint does not contain a valid optimizer state dictionary"
            )

        saved_state = optimizer_state.get("state")
        if not isinstance(saved_state, dict):
            raise ValueError(
                "Checkpoint optimizer state entries are invalid"
            )

        saved_groups = optimizer_state.get("param_groups")
        if not isinstance(saved_groups, list):
            raise ValueError(
                "Checkpoint optimizer parameter groups are invalid"
            )

        live_groups = optimizer.param_groups
        if len(saved_groups) != len(live_groups):
            raise ValueError(
                "Checkpoint optimizer parameter-group count does not match"
            )

        parameter_map = {}
        for saved_group, live_group in zip(
            saved_groups, live_groups, strict=True
        ):
            if not isinstance(saved_group, dict):
                raise TypeError(
                    "Checkpoint optimizer parameter group is invalid"
                )

            saved_ids = saved_group.get("params")
            live_parameters = live_group.get("params")

            if (
                not isinstance(saved_ids, list)
                or len(saved_ids) != len(live_parameters)
            ):
                raise ValueError(
                    "Checkpoint optimizer parameter-group layout does not match"
                )

            for saved_id, parameter in zip(
                saved_ids, live_parameters, strict=True
            ):
                if (
                    isinstance(saved_id, bool)
                    or not isinstance(saved_id, int)
                    or saved_id in parameter_map
                ):
                    raise ValueError(
                        "Checkpoint optimizer parameter IDs are invalid"
                    )
                parameter_map[saved_id] = parameter

        if not saved_state.keys() <= parameter_map.keys():
            raise ValueError(
                "Checkpoint optimizer state references unknown parameters"
            )

        for parameter_id, parameter_state in saved_state.items():
            if not isinstance(parameter_state, dict):
                raise TypeError(
                    "Checkpoint optimizer parameter state must be a dictionary"
                )

            parameter = parameter_map[parameter_id]
            for state_name, value in parameter_state.items():
                if torch.is_tensor(value):
                    if state_name == "step":
                        if value.numel() != 1:
                            raise ValueError(
                                "Checkpoint optimizer step state must be scalar"
                            )
                    elif value.shape != parameter.shape:
                        raise ValueError(
                            "Checkpoint optimizer state shape mismatch "
                            f"for parameter {parameter_id!r}, "
                            f"field {state_name!r}"
                        )
                elif not isinstance(value, (int, float, bool)):
                    raise TypeError(
                        "Checkpoint optimizer state contains an unsupported value"
                    )

    if scaler is not None and "grad_scaler_state_dict" in checkpoint:
        saved_scaler = checkpoint["grad_scaler_state_dict"]
        if not isinstance(saved_scaler, dict):
            raise TypeError(
                "Checkpoint gradient-scaler state must be a dictionary"
            )

        import copy

        try:
            copy.deepcopy(scaler).load_state_dict(saved_scaler)
        except Exception as exc:
            raise ValueError(
                "Checkpoint gradient-scaler state is invalid"
            ) from exc

    rng_keys = {
        "torch_rng_state",
        "cuda_rng_state_all",
        "python_rng_state",
        "numpy_rng_state",
    }
    _validate_rng_state({
        key: value
        for key, value in checkpoint.items()
        if key in rng_keys
    })


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    scaler: Any | None = None,
    expected_signature: dict[str, Any] | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore checkpoint state, optionally deferring RNG restoration."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to load checkpoint {checkpoint_path}. "
            "The file may be corrupted or incomplete. Preserve the file "
            "for diagnosis and resume from a known-good checkpoint or "
            "start a new run."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint contents must be a dictionary")

    required_keys = {"step", "model_state_dict"}
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required fields: {sorted(missing_keys)}"
        )

    step = checkpoint["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TypeError("Checkpoint step must be a non-negative integer")

    best_validation_loss = checkpoint.get("best_validation_loss")
    if best_validation_loss is not None and (
        isinstance(best_validation_loss, bool)
        or not isinstance(best_validation_loss, (int, float))
        or math.isnan(best_validation_loss)
        or best_validation_loss == -math.inf
    ):
        raise ValueError(
            "Checkpoint best_validation_loss must be a real number "
            "other than NaN or negative infinity, or None"
        )

    batches_in_current_pass = checkpoint.get(
        "batches_in_current_pass", 0
    )
    if (
        isinstance(batches_in_current_pass, bool)
        or not isinstance(batches_in_current_pass, int)
        or batches_in_current_pass < 0
    ):
        raise ValueError(
            "Checkpoint batches_in_current_pass must be a "
            "non-negative integer"
        )

    if expected_signature is not None:
        _validate_resume_signature(
            expected_signature,
            "expected_signature",
        )
        saved_signature = checkpoint.get("resume_signature")

        if saved_signature is None:
            import warnings

            warnings.warn(
                "Legacy checkpoint has no resume signature; compatibility "
                "with the current training data and configuration "
                "cannot be verified.",
                UserWarning,
                stacklevel=2,
            )
        else:
            _validate_resume_signature(
                saved_signature,
                "checkpoint resume_signature",
            )
            if saved_signature != expected_signature:
                raise ValueError(
                    "Checkpoint resume signature does not match the "
                    "current training data, tokenizer, or batch "
                    "configuration. Use the original inputs or start "
                    "a new checkpoint run."
                )

        if saved_signature is not None:
            required_rng_keys = {
                "torch_rng_state",
                "python_rng_state",
                "numpy_rng_state",
            }
            if not required_rng_keys <= checkpoint.keys():
                raise ValueError(
                    "Signed checkpoint is missing required RNG states"
                )

    _validate_checkpoint_restore_state(
        checkpoint,
        model,
        optimizer,
        scaler,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "grad_scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

    rng_keys = {
        "torch_rng_state",
        "cuda_rng_state_all",
        "python_rng_state",
        "numpy_rng_state",
    }
    rng_state = {
        key: value
        for key, value in checkpoint.items()
        if key in rng_keys
    }

    if restore_rng:
        restore_rng_state(rng_state)

    metadata = {
        "step": step,
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "batches_in_current_pass": batches_in_current_pass,
    }

    if not restore_rng:
        metadata["rng_state"] = rng_state

    return metadata