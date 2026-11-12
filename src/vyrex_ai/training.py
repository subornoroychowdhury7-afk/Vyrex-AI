"""Training utilities for VYREX AI."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return minimum_lr + cosine * (peak_lr - minimum_lr)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_validation_loss: float | None = None,
    scaler: Any | None = None,
) -> None:
    """Save a checkpoint atomically to avoid incomplete checkpoint files."""
    if step < 0:
        raise ValueError("Checkpoint step cannot be negative")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_validation_loss,
        "torch_rng_state": torch.get_rng_state().cpu(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    if scaler is not None and scaler.is_enabled():
        checkpoint["grad_scaler_state_dict"] = scaler.state_dict()

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = [
            state.cpu() for state in torch.cuda.get_rng_state_all()
        ]

    temporary_path = checkpoint_path.with_name(
        checkpoint_path.name + ".tmp"
    )

    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Restore model, optional optimizer, and random-number-generator states."""
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    required_keys = {"step", "model_state_dict"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required fields: {sorted(missing_keys)}"
        )

    step = checkpoint["step"]
    if not isinstance(step, int) or step < 0:
        raise ValueError("Checkpoint step must be a non-negative integer")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "grad_scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        saved_cuda_states = checkpoint["cuda_rng_state_all"]
        available_devices = torch.cuda.device_count()
        states_to_restore = [
            state.cpu() for state in saved_cuda_states[:available_devices]
        ]

        if states_to_restore:
            torch.cuda.set_rng_state_all(states_to_restore)

    return {
        "step": step,
        "best_validation_loss": checkpoint.get("best_validation_loss"),
    }
