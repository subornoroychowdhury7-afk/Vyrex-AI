"""Training and validation loop for VYREX AI."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vyrex_ai.training import (
    learning_rate_at_step,
    load_checkpoint,
    save_checkpoint,
)


def _get_tokens(batch: Any) -> torch.Tensor:
    """Extract token IDs from a supported batch format."""
    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise ValueError("Dictionary batches must contain 'input_ids'")
        batch = batch["input_ids"]

    if not isinstance(batch, torch.Tensor):
        raise TypeError("Each batch must be a Tensor or contain input_ids")

    if batch.ndim != 2 or batch.size(1) < 2:
        raise ValueError(
            "Batches must have shape [batch, sequence] with length >= 2"
        )

    if batch.dtype != torch.long:
        batch = batch.long()

    return batch


def _resolve_precision(
    device: torch.device,
    precision: str,
) -> torch.dtype | None:
    """Resolve and validate the requested training precision."""
    if precision == "fp32":
        return None

    if precision not in {"fp16", "bf16"}:
        raise ValueError(
            f"Unsupported precision: {precision!r}. "
            "Expected fp32, fp16, or bf16."
        )

    if device.type != "cuda":
        raise RuntimeError(
            f"{precision} precision requires CUDA in this training implementation"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if precision == "fp16":
        return torch.float16

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 precision is not supported by this CUDA device")

    return torch.bfloat16


def _autocast(
    device: torch.device,
    dtype: torch.dtype | None,
):
    """Return an autocast context for the selected precision."""
    if dtype is None:
        return nullcontext()

    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
    )


def _evaluate(
    model: nn.Module,
    validation_loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    """Calculate mean next-token validation loss."""
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.inference_mode():
        for batch in validation_loader:
            tokens = _get_tokens(batch).to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            with _autocast(device, autocast_dtype):
                _, loss = model(inputs, targets)

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Validation produced an invalid loss")

            total_loss += loss.item()
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                break

    if batch_count == 0:
        raise ValueError("Validation loader produced no batches")

    return total_loss / batch_count


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    training_config: Any,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
    max_validation_batches: int | None = None,
) -> dict[str, Any]:
    """Train a causal language model and periodically evaluate/checkpoint it."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    precision = getattr(training_config, "precision", "fp32")
    autocast_dtype = _resolve_precision(device, precision)

    scaler = None
    if precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=tuple(training_config.betas),
        weight_decay=training_config.weight_decay,
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_validation_loss = float("inf")

    if resume_from is not None:
        metadata = load_checkpoint(
            resume_from,
            model,
            optimizer,
            map_location=device,
            scaler=scaler,
        )
        start_step = metadata["step"]
        saved_best = metadata["best_validation_loss"]
        if saved_best is not None:
            best_validation_loss = saved_best

    if start_step >= training_config.max_steps:
        raise ValueError("Checkpoint step has already reached max_steps")

    train_iterator = iter(train_loader)
    accumulation_steps = training_config.gradient_accumulation_steps
    log_interval = training_config.log_every_steps
    eval_interval = training_config.eval_every_steps
    checkpoint_interval = training_config.checkpoint_every_steps
    max_steps = training_config.max_steps
    grad_clip = training_config.gradient_clip_norm

    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    history: list[dict[str, float | int]] = []
    model.train()

    for step in range(start_step, max_steps):
        current_step = step + 1
        learning_rate = learning_rate_at_step(step, training_config)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                try:
                    batch = next(train_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "Training loader produced no batches"
                    ) from exc

            tokens = _get_tokens(batch).to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            with _autocast(device, autocast_dtype):
                _, loss = model(inputs, targets)

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training produced an invalid loss")

            scaled_loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accumulated_loss += loss.detach().item()

        if scaler is not None and grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        mean_loss = accumulated_loss / accumulation_steps

        if log_interval > 0 and current_step % log_interval == 0:
            print(
                f"Step {current_step}/{max_steps} | "
                f"train_loss={mean_loss:.4f} | lr={learning_rate:.7g}"
            )

        if eval_interval > 0 and current_step % eval_interval == 0:
            validation_loss = _evaluate(
                model,
                validation_loader,
                device,
                max_validation_batches,
                autocast_dtype,
            )

            print(
                f"Validation at step {current_step}: "
                f"loss={validation_loss:.4f}"
            )

            history.append({
                "step": current_step,
                "train_loss": mean_loss,
                "validation_loss": validation_loss,
            })

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    current_step,
                    best_validation_loss,
                    scaler=scaler,
                )

            model.train()

        if (
            checkpoint_interval > 0
            and current_step % checkpoint_interval == 0
        ):
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                current_step,
                best_validation_loss,
                scaler=scaler,
            )

    save_checkpoint(
        checkpoint_dir / "latest.pt",
        model,
        optimizer,
        max_steps,
        (
            best_validation_loss
            if best_validation_loss != float("inf")
            else None
        ),
        scaler=scaler,
    )

    return {
        "completed_steps": max_steps,
        "best_validation_loss": (
            best_validation_loss
            if best_validation_loss != float("inf")
            else None
        ),
        "history": history,
        "device": str(device),
        "precision": precision,
    }
