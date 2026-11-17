"""Training and validation loop for VYREX AI."""

from __future__ import annotations

import csv

from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vyrex_ai.training import (
    learning_rate_at_step,
    load_checkpoint,
    restore_rng_state,
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
    total_tokens = 0
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

            token_count = targets.numel()
            if token_count <= 0:
                raise RuntimeError("Validation batch contains no target tokens")

            total_loss += loss.detach().item() * token_count
            total_tokens += token_count
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                break

    if total_tokens == 0:
        raise ValueError("Validation loader produced no batches")

    return total_loss / total_tokens



def _append_training_metric(
    path: Path,
    step: int,
    train_loss: float,
    learning_rate: float,
    validation_loss: float | None,
) -> None:
    """Append one training metric row without loading prior history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "train_loss", "lr", "validation_loss"],
        )
        if write_header:
            writer.writeheader()

        writer.writerow({
            "step": step,
            "train_loss": f"{train_loss:.8f}",
            "lr": f"{learning_rate:.10g}",
            "validation_loss": (
                "" if validation_loss is None
                else f"{validation_loss:.8f}"
            ),
        })


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    training_config: Any,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
    max_validation_batches: int | None = None,
    resume_signature: dict[str, Any] | None = None,
    metrics_path: str | Path | None = None,
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
    metrics_path = Path(metrics_path) if metrics_path is not None else None

    start_step = 0
    best_validation_loss = float("inf")
    batches_in_current_pass = 0

    if resume_from is not None:
        metadata = load_checkpoint(
            resume_from,
            model,
            optimizer,
            map_location=device,
            scaler=scaler,
            expected_signature=resume_signature,
            restore_rng=False,
        )
        start_step = metadata["step"]
        batches_in_current_pass = metadata["batches_in_current_pass"]
        saved_best = metadata["best_validation_loss"]
        if saved_best is not None:
            best_validation_loss = saved_best

    if start_step >= training_config.max_steps:
        raise ValueError("Checkpoint step has already reached max_steps")

    train_iterator = iter(train_loader)

    # Reconstruct the saved position in the current deterministic data pass.
    for _ in range(batches_in_current_pass):
        try:
            next(train_iterator)
        except StopIteration as exc:
            raise ValueError(
                "Checkpoint data position exceeds the current training "
                "loader. The dataset or batch configuration may have changed."
            ) from exc

    if resume_from is not None:
        restore_rng_state(metadata["rng_state"])

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
                batches_in_current_pass = 0
                try:
                    batch = next(train_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "Training loader produced no batches"
                    ) from exc

            batches_in_current_pass += 1
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

        should_log = log_interval > 0 and current_step % log_interval == 0
        should_eval = eval_interval > 0 and current_step % eval_interval == 0
        validation_loss = None

        if should_log:
            print(
                f"Step {current_step}/{max_steps} | "
                f"train_loss={mean_loss:.4f} | lr={learning_rate:.7g}"
            )

        if should_eval:
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
                    batches_in_current_pass=batches_in_current_pass,
                    resume_signature=resume_signature,
                )

            model.train()

        if metrics_path is not None and (should_log or should_eval):
            _append_training_metric(
                metrics_path,
                current_step,
                mean_loss,
                learning_rate,
                validation_loss,
            )

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
                batches_in_current_pass=batches_in_current_pass,
                resume_signature=resume_signature,
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
        batches_in_current_pass=batches_in_current_pass,
        resume_signature=resume_signature,
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
