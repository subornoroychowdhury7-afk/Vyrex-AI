"""Progressive five-stage training with shared optimizer and recovery."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from vyrex_ai.config import EXPECTED_CURRICULUM_STAGES
from vyrex_ai.trainer import (
    _autocast,
    _evaluate,
    _get_tokens,
    _resolve_precision,
)
from vyrex_ai.training import (
    capture_rng_state,
    learning_rate_at_step,
    restore_rng_state,
)


def _stage_step_counts(total_steps: int) -> tuple[int, ...]:
    """Distribute all optimizer steps across the five stages."""
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TypeError("max_steps must be an integer")
    if total_steps < len(EXPECTED_CURRICULUM_STAGES):
        raise ValueError("max_steps must allocate at least one step per stage")

    base, remainder = divmod(total_steps, len(EXPECTED_CURRICULUM_STAGES))
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(len(EXPECTED_CURRICULUM_STAGES))
    )


def _save_curriculum_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    scaler: Any | None,
) -> None:
    """Atomically save complete curriculum recovery state."""
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "curriculum_state": state,
        "rng_state": capture_rng_state(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        with temporary.open("wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_curriculum_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_signature: dict[str, Any],
    device: torch.device,
    scaler: Any | None,
) -> dict[str, Any]:
    """Validate compatibility before restoring curriculum state."""
    if not path.is_file():
        raise FileNotFoundError(f"Curriculum checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported or malformed curriculum checkpoint")

    state = checkpoint.get("curriculum_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no valid curriculum state")

    if state.get("signature") != expected_signature:
        raise ValueError(
            "Curriculum resume signature mismatch. Dataset, tokenizer, "
            "model, or training configuration may have changed."
        )

    required = {
        "active_stage_index",
        "stage_step",
        "global_step",
        "batches_in_current_pass",
        "completed_stages",
        "history",
        "best_validation_loss",
        "signature",
    }
    if not required.issubset(state):
        raise ValueError("Curriculum checkpoint state is incomplete")

    index = state["active_stage_index"]
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index <= len(EXPECTED_CURRICULUM_STAGES)
    ):
        raise ValueError("Invalid active curriculum stage index")

    for key in ("stage_step", "global_step", "batches_in_current_pass"):
        value = state[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid curriculum checkpoint field: {key}")

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint model state is invalid")

    current_state = model.state_dict()
    if model_state.keys() != current_state.keys():
        raise ValueError("Checkpoint model parameters do not match this model")

    for key, current_tensor in current_state.items():
        saved_tensor = model_state[key]
        if (
            not isinstance(saved_tensor, torch.Tensor)
            or saved_tensor.shape != current_tensor.shape
        ):
            raise ValueError(f"Invalid checkpoint model tensor: {key}")

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint optimizer state is invalid")

    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        raise ValueError("Checkpoint RNG state is invalid")

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and not isinstance(scaler_state, dict):
        raise ValueError("Checkpoint is missing mixed-precision scaler state")
    if scaler is None and scaler_state is not None:
        raise ValueError("Checkpoint precision does not match this run")

    # Restore only after the compatibility checks above.
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

    if scaler is not None:
        scaler.load_state_dict(scaler_state)

    state["_rng_state"] = rng_state
    return state


def train_curriculum(
    model: nn.Module,
    stage_loaders: Mapping[str, tuple[Any, Any]],
    training_config: Any,
    checkpoint_dir: str | Path,
    resume_signature: dict[str, Any],
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
    max_validation_batches: int | None = None,
) -> dict[str, Any]:
    """Train all five cumulative stages with one persistent optimizer.

    stage_loaders maps each approved stage to (train_loader, validation_loader).
    Resume only from latest.pt. best.pt is a reporting artifact, not a
    continuation checkpoint.
    """
    if tuple(stage_loaders.keys()) != EXPECTED_CURRICULUM_STAGES:
        raise ValueError(
            "stage_loaders must contain the five approved stages in order"
        )

    if not isinstance(resume_signature, dict) or not resume_signature:
        raise ValueError("A non-empty curriculum resume signature is required")

    if (
        max_validation_batches is not None
        and (
            isinstance(max_validation_batches, bool)
            or not isinstance(max_validation_batches, int)
            or max_validation_batches < 1
        )
    ):
        raise ValueError("max_validation_batches must be a positive integer")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    precision = getattr(training_config, "precision", "fp32")
    autocast_dtype = _resolve_precision(device, precision)

    scaler = None
    if precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    stage_steps = _stage_step_counts(training_config.max_steps)
    accumulation_steps = training_config.gradient_accumulation_steps
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    # Preflight every stage before model or checkpoint creation.
    for stage in EXPECTED_CURRICULUM_STAGES:
        train_loader, validation_loader = stage_loaders[stage]
        if next(iter(train_loader), None) is None:
            raise ValueError(f"{stage}: training split produces no token blocks")
        if next(iter(validation_loader), None) is None:
            raise ValueError(f"{stage}: validation split produces no token blocks")

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=tuple(training_config.betas),
        weight_decay=training_config.weight_decay,
    )

    checkpoint_directory = Path(checkpoint_dir)
    latest_path = checkpoint_directory / "latest.pt"
    best_path = checkpoint_directory / "best.pt"

    state: dict[str, Any] = {
        "active_stage_index": 0,
        "stage_step": 0,
        "global_step": 0,
        "batches_in_current_pass": 0,
        "completed_stages": [],
        "history": [],
        "best_validation_loss": None,
        "signature": resume_signature,
    }

    if resume_from is not None:
        resume_path = Path(resume_from).resolve()
        expected_latest = latest_path.resolve()
        if resume_path != expected_latest:
            raise ValueError("Curriculum training may resume only from latest.pt")

        state = _load_curriculum_checkpoint(
            resume_path,
            model,
            optimizer,
            resume_signature,
            device,
            scaler,
        )

        if state["global_step"] > training_config.max_steps:
            raise ValueError("Checkpoint global step exceeds configured max_steps")

        index = state["active_stage_index"]
        if index < len(EXPECTED_CURRICULUM_STAGES):
            target = stage_steps[index]
            if state["stage_step"] > target:
                raise ValueError("Checkpoint stage position is inconsistent")
            if state["stage_step"] == target:
                history = state["history"]
                expected_stage = EXPECTED_CURRICULUM_STAGES[index]
                if (
                    not isinstance(history, list)
                    or not history
                    or not isinstance(history[-1], dict)
                    or history[-1].get("stage") != expected_stage
                    or history[-1].get("stage_step") != target
                    or history[-1].get("global_step") != state["global_step"]
                ):
                    raise ValueError(
                        "Final-step checkpoint is missing valid stage validation"
                    )
        elif state["global_step"] != training_config.max_steps:
            raise ValueError("Completed curriculum checkpoint has invalid step count")

    else:
        if checkpoint_directory.exists() and any(checkpoint_directory.iterdir()):
            raise FileExistsError(
                "Checkpoint directory is not empty. Supply --resume with "
                "its latest.pt or choose a new directory."
            )

    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    def save_latest() -> None:
        _save_curriculum_checkpoint(
            latest_path, model, optimizer, state, scaler
        )

    # Restore RNG only after the resumed stage iterator has been
    # constructed and advanced to its saved position.
    rng_state_pending = resume_from is not None

    log_interval = training_config.log_every_steps
    eval_interval = training_config.eval_every_steps
    checkpoint_interval = training_config.checkpoint_every_steps
    grad_clip = training_config.gradient_clip_norm

    while state["active_stage_index"] < len(EXPECTED_CURRICULUM_STAGES):
        stage_index = state["active_stage_index"]
        stage = EXPECTED_CURRICULUM_STAGES[stage_index]
        train_loader, validation_loader = stage_loaders[stage]
        target_steps = stage_steps[stage_index]

        train_iterator = iter(train_loader)
        for _ in range(state["batches_in_current_pass"]):
            try:
                next(train_iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"{stage}: saved data position exceeds current loader"
                ) from exc

        if rng_state_pending:
            restore_rng_state(state.pop("_rng_state"))
            rng_state_pending = False

        model.train()
        last_evaluation_step = (
            target_steps if state["stage_step"] == target_steps else -1
        )

        while state["stage_step"] < target_steps:
            global_step = state["global_step"]
            learning_rate = learning_rate_at_step(
                global_step, training_config
            )

            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            for _ in range(accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    state["batches_in_current_pass"] = 0
                    try:
                        batch = next(train_iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            f"{stage}: training loader produced no batches"
                        ) from exc

                state["batches_in_current_pass"] += 1
                tokens = _get_tokens(batch).to(device)
                inputs, targets = tokens[:, :-1], tokens[:, 1:]

                with _autocast(device, autocast_dtype):
                    _, loss = model(inputs, targets)

                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"{stage}: training produced invalid loss")

                scaled_loss = loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accumulated_loss += loss.detach().item()

            if scaler is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            state["stage_step"] += 1
            state["global_step"] += 1
            mean_loss = accumulated_loss / accumulation_steps

            if log_interval > 0 and state["global_step"] % log_interval == 0:
                print(
                    f"{stage} | step {state['global_step']}/"
                    f"{training_config.max_steps} | "
                    f"train_loss={mean_loss:.4f} | lr={learning_rate:.7g}"
                )

            should_evaluate = (
                eval_interval > 0
                and state["global_step"] % eval_interval == 0
            ) or state["stage_step"] == target_steps

            if should_evaluate:
                validation_loss = _evaluate(
                    model,
                    validation_loader,
                    device,
                    max_validation_batches,
                    autocast_dtype,
                )
                if not torch.isfinite(torch.tensor(validation_loss)):
                    raise RuntimeError(f"{stage}: invalid validation loss")

                state["history"].append({
                    "stage": stage,
                    "stage_step": state["stage_step"],
                    "global_step": state["global_step"],
                    "train_loss": mean_loss,
                    "validation_loss": validation_loss,
                })
                last_evaluation_step = state["stage_step"]

                previous_best = state["best_validation_loss"]
                if previous_best is None or validation_loss < previous_best:
                    state["best_validation_loss"] = validation_loss
                    _save_curriculum_checkpoint(
                        best_path, model, optimizer, state, scaler
                    )

                print(
                    f"{stage} validation | global_step="
                    f"{state['global_step']} | loss={validation_loss:.4f}"
                )
                model.train()

            if (
                checkpoint_interval > 0
                and state["global_step"] % checkpoint_interval == 0
            ):
                save_latest()

        # Mandatory final validation has occurred at the final stage step.
        if last_evaluation_step != target_steps:
            raise RuntimeError(f"{stage}: final stage validation was not completed")

        state["completed_stages"].append(stage)
        state["active_stage_index"] += 1
        state["stage_step"] = 0
        state["batches_in_current_pass"] = 0

        # Persist the stage transition before beginning the next stage.
        save_latest()
        print(f"CURRICULUM STAGE COMPLETE: {stage}")

    if state["global_step"] != training_config.max_steps:
        raise RuntimeError("Curriculum ended at an unexpected global step")

    return {
        "completed_steps": state["global_step"],
        "completed_stages": list(state["completed_stages"]),
        "best_validation_loss": state["best_validation_loss"],
        "history": list(state["history"]),
        "device": str(device),
        "precision": precision,
        "checkpoint": str(latest_path),
    }
