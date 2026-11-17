import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vyrex_ai.training import (
    build_optimizer,
    learning_rate_at_step,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


@pytest.fixture
def training_config():
    return SimpleNamespace(
        learning_rate=0.001,
        min_learning_rate=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        warmup_steps=2,
        max_steps=10,
    )


@pytest.fixture
def model():
    torch.manual_seed(42)
    return nn.Linear(4, 2)


def test_build_optimizer(model, training_config):
    optimizer = build_optimizer(model, training_config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_learning_rate_warmup(training_config):
    assert learning_rate_at_step(0, training_config) == pytest.approx(0.0005)
    assert learning_rate_at_step(1, training_config) == pytest.approx(0.001)


def test_learning_rate_cosine_decay(training_config):
    assert learning_rate_at_step(10, training_config) == pytest.approx(0.0001)
    assert learning_rate_at_step(100, training_config) == pytest.approx(0.0001)


def test_checkpoint_save_and_restore(tmp_path, model, training_config):
    optimizer = build_optimizer(model, training_config)
    checkpoint_path = tmp_path / "checkpoint.pt"

    # Populate AdamW state before saving so restoration tests include
    # optimizer step counters and both moment estimates.
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()

    original_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=25,
        best_validation_loss=1.234,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()

    metadata = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        map_location="cpu",
    )

    assert metadata["step"] == 25
    assert metadata["best_validation_loss"] == pytest.approx(1.234)
    assert metadata["batches_in_current_pass"] == 0

    for restored, original in zip(model.parameters(), original_parameters):
        assert torch.equal(restored, original)

    restored_optimizer_state = optimizer.state_dict()
    assert restored_optimizer_state["param_groups"] == (
        original_optimizer_state["param_groups"]
    )
    for parameter_id, expected_state in original_optimizer_state["state"].items():
        actual_state = restored_optimizer_state["state"][parameter_id]
        assert actual_state["step"] == expected_state["step"]
        assert torch.equal(actual_state["exp_avg"], expected_state["exp_avg"])
        assert torch.equal(
            actual_state["exp_avg_sq"],
            expected_state["exp_avg_sq"],
        )


def test_checkpoint_restores_random_states(tmp_path, model, training_config):
    optimizer = build_optimizer(model, training_config)
    checkpoint_path = tmp_path / "rng_checkpoint.pt"

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    save_checkpoint(checkpoint_path, model, optimizer, step=3)

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)

    random.random()
    np.random.random()
    torch.rand(1)

    load_checkpoint(checkpoint_path, model, optimizer, map_location="cpu")

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_deferred_rng_restore_survives_dataloader_iterator(
    tmp_path, model, training_config
):
    optimizer = build_optimizer(model, training_config)
    checkpoint_path = tmp_path / "deferred_rng.pt"

    random.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    save_checkpoint(checkpoint_path, model, optimizer, step=1)
    expected = torch.rand(1)

    torch.manual_seed(999)
    metadata = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        restore_rng=False,
    )

    loader = torch.utils.data.DataLoader(
        [torch.tensor([1]), torch.tensor([2])],
        batch_size=1,
        num_workers=0,
    )
    iterator = iter(loader)
    next(iterator)

    restore_rng_state(metadata["rng_state"])
    assert torch.equal(torch.rand(1), expected)


def test_checkpoint_rejects_negative_step(tmp_path, model, training_config):
    optimizer = build_optimizer(model, training_config)

    with pytest.raises(ValueError, match="cannot be negative"):
        save_checkpoint(tmp_path / "invalid.pt", model, optimizer, step=-1)


def test_missing_checkpoint_raises(tmp_path, model):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)


def test_checkpoint_requires_optimizer_state_when_optimizer_given(
    tmp_path, model, training_config
):
    optimizer = build_optimizer(model, training_config)
    checkpoint_path = tmp_path / "without_optimizer.pt"

    torch.save(
        {"step": 1, "model_state_dict": model.state_dict()},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="optimizer state"):
        load_checkpoint(checkpoint_path, model, optimizer)

def test_checkpoint_round_trips_data_position(
    tmp_path, model, training_config
):
    optimizer = build_optimizer(model, training_config)
    checkpoint_path = tmp_path / "position.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=4,
        batches_in_current_pass=7,
    )

    metadata = load_checkpoint(checkpoint_path, model, optimizer)

    assert metadata["batches_in_current_pass"] == 7


def test_legacy_checkpoint_defaults_data_position_to_zero(
    tmp_path, model
):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {"step": 2, "model_state_dict": model.state_dict()},
        checkpoint_path,
    )

    metadata = load_checkpoint(checkpoint_path, model)

    assert metadata["batches_in_current_pass"] == 0


@pytest.mark.parametrize("position", [-1, 1.5, True])
def test_checkpoint_rejects_invalid_data_position(
    tmp_path, model, training_config, position
):
    optimizer = build_optimizer(model, training_config)

    with pytest.raises(ValueError, match="batches_in_current_pass"):
        save_checkpoint(
            tmp_path / "invalid_position.pt",
            model,
            optimizer,
            step=1,
            batches_in_current_pass=position,
        )

def test_checkpoint_round_trips_resume_signature(
    tmp_path, model, training_config
):
    checkpoint_path = tmp_path / "signed.pt"
    signature = {
        "train_data_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "sequence_length": 128,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "validation_data_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "training_config_sha256": "e" * 64,
    }

    save_checkpoint(
        checkpoint_path,
        model,
        build_optimizer(model, training_config),
        step=3,
        resume_signature=signature,
    )

    metadata = load_checkpoint(
        checkpoint_path,
        model,
        expected_signature=signature,
    )

    assert metadata["step"] == 3


def test_checkpoint_rejects_mismatched_signature_before_restore(
    tmp_path, model, training_config
):
    checkpoint_path = tmp_path / "signed.pt"
    saved_signature = {
        "train_data_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "sequence_length": 128,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "validation_data_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "training_config_sha256": "e" * 64,
    }
    expected_signature = {
        **saved_signature,
        "train_data_sha256": "c" * 64,
    }

    save_checkpoint(
        checkpoint_path,
        model,
        build_optimizer(model, training_config),
        step=3,
        resume_signature=saved_signature,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    parameters_before_load = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    with pytest.raises(ValueError, match="signature"):
        load_checkpoint(
            checkpoint_path,
            model,
            expected_signature=expected_signature,
        )

    for parameter, expected in zip(
        model.parameters(), parameters_before_load, strict=True
    ):
        assert torch.equal(parameter, expected)


def test_legacy_checkpoint_warns_but_remains_loadable(
    tmp_path, model
):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {"step": 2, "model_state_dict": model.state_dict()},
        checkpoint_path,
    )
    expected_signature = {
        "train_data_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "sequence_length": 128,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "validation_data_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "training_config_sha256": "e" * 64,
    }

    with pytest.warns(UserWarning, match="Legacy checkpoint"):
        metadata = load_checkpoint(
            checkpoint_path,
            model,
            expected_signature=expected_signature,
        )

    assert metadata["step"] == 2

def _valid_resume_signature():
    return {
        "train_data_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "sequence_length": 128,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "validation_data_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "training_config_sha256": "e" * 64,
    }


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("train_data_sha256", "not-a-hash", "SHA-256"),
        ("sequence_length", True, "positive integer"),
        ("micro_batch_size", 0, "positive integer"),
    ],
)
def test_checkpoint_rejects_malformed_resume_signature_on_save(
    tmp_path, model, training_config, field, invalid_value, message
):
    signature = _valid_resume_signature()
    signature[field] = invalid_value
    optimizer = build_optimizer(model, training_config)

    with pytest.raises(ValueError, match=message):
        save_checkpoint(
            tmp_path / "malformed.pt",
            model,
            optimizer,
            step=1,
            resume_signature=signature,
        )


def test_checkpoint_rejects_resume_signature_with_missing_field_on_save(
    tmp_path, model, training_config
):
    signature = _valid_resume_signature()
    del signature["tokenizer_sha256"]
    optimizer = build_optimizer(model, training_config)

    with pytest.raises(ValueError, match="exactly these fields"):
        save_checkpoint(
            tmp_path / "missing-field.pt",
            model,
            optimizer,
            step=1,
            resume_signature=signature,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("tokenizer_sha256", "invalid", "SHA-256"),
        ("gradient_accumulation_steps", False, "positive integer"),
        ("sequence_length", -1, "positive integer"),
    ],
)
def test_checkpoint_rejects_malformed_saved_resume_signature(
    tmp_path, model, field, invalid_value, message
):
    signature = _valid_resume_signature()
    signature[field] = invalid_value
    checkpoint_path = tmp_path / "malformed-saved-signature.pt"

    torch.save(
        {
            "step": 1,
            "model_state_dict": model.state_dict(),
            "resume_signature": signature,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match=message):
        load_checkpoint(
            checkpoint_path,
            model,
            expected_signature=_valid_resume_signature(),
        )



def test_checkpoint_save_failure_preserves_previous_file(
    tmp_path, model, training_config, monkeypatch
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    original_contents = b"previous-valid-checkpoint"
    checkpoint_path.write_bytes(original_contents)

    def fail_after_partial_write(checkpoint, path):
        with open(path, "wb") as handle:
            handle.write(b"partial-checkpoint")
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated checkpoint write failure"):
        save_checkpoint(
            checkpoint_path,
            model,
            build_optimizer(model, training_config),
            step=5,
        )

    assert checkpoint_path.read_bytes() == original_contents
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []

def test_truncated_checkpoint_has_clear_corruption_error(tmp_path, model):
    path = tmp_path / "truncated.pt"
    path.write_bytes(b"not-a-complete-pytorch-checkpoint")
    with pytest.raises(ValueError, match="corrupted or incomplete"):
        load_checkpoint(path, model)


def test_checkpoint_rejects_non_dictionary_contents(tmp_path, model):
    path = tmp_path / "invalid.pt"
    torch.save(["invalid"], path)
    with pytest.raises(TypeError, match="contents must be a dictionary"):
        load_checkpoint(path, model)


def test_invalid_model_state_is_rejected_before_model_mutation(tmp_path, model):
    path = tmp_path / "invalid-model.pt"
    original = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    invalid = {key: value.clone() for key, value in original.items()}
    invalid.pop(next(iter(invalid)))
    torch.save({"step": 1, "model_state_dict": invalid}, path)

    with pytest.raises(ValueError, match="keys do not match"):
        load_checkpoint(path, model)

    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


def test_missing_optimizer_state_is_rejected_before_model_mutation(
    tmp_path, model, training_config
):
    path = tmp_path / "missing-optimizer.pt"
    optimizer = build_optimizer(model, training_config)
    torch.save(
        {"step": 1, "model_state_dict": model.state_dict()},
        path,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)

    mutated = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="optimizer state"):
        load_checkpoint(path, model, optimizer)

    for key, value in model.state_dict().items():
        assert torch.equal(value, mutated[key])

def test_malformed_optimizer_moments_rejected_before_model_mutation(
    tmp_path, model, training_config
):
    optimizer = build_optimizer(model, training_config)
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()

    path = tmp_path / "bad-optimizer-moments.pt"
    save_checkpoint(path, model, optimizer, step=1)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    parameter_id = next(iter(saved["optimizer_state_dict"]["state"]))
    saved["optimizer_state_dict"]["state"][parameter_id]["exp_avg"] = torch.zeros(1)
    torch.save(saved, path)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2)

    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="optimizer state shape mismatch"):
        load_checkpoint(path, model, optimizer)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_invalid_rng_state_rejected_before_model_mutation(
    tmp_path, model, training_config
):
    optimizer = build_optimizer(model, training_config)
    path = tmp_path / "bad-rng.pt"
    save_checkpoint(path, model, optimizer, step=1)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved["python_rng_state"] = "invalid-rng-state"
    torch.save(saved, path)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3)

    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="Python RNG state is invalid"):
        load_checkpoint(path, model, optimizer)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize(
    "missing_key",
    ["torch_rng_state", "numpy_rng_state", "python_rng_state"],
)
def test_partial_rng_state_rejected_before_model_mutation(
    tmp_path, model, training_config, missing_key
):
    optimizer = build_optimizer(model, training_config)
    path = tmp_path / "partial-rng.pt"
    save_checkpoint(path, model, optimizer, step=1)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved.pop(missing_key)
    torch.save(saved, path)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(4)

    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="incomplete"):
        load_checkpoint(path, model, optimizer)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("invalid_step", [True, 1.5, "2"])
def test_checkpoint_rejects_non_integer_step(
    tmp_path, model, training_config, invalid_step
):
    optimizer = build_optimizer(model, training_config)
    with pytest.raises(TypeError, match="step must be a non-negative integer"):
        save_checkpoint(
            tmp_path / "invalid-step.pt",
            model,
            optimizer,
            step=invalid_step,
        )


@pytest.mark.parametrize("invalid_loss", [float("nan"), float("-inf"), True])
def test_checkpoint_rejects_invalid_best_validation_loss(
    tmp_path, model, training_config, invalid_loss
):
    optimizer = build_optimizer(model, training_config)
    with pytest.raises(ValueError, match="best_validation_loss must be a real number"):
        save_checkpoint(
            tmp_path / "invalid-loss.pt",
            model,
            optimizer,
            step=1,
            best_validation_loss=invalid_loss,
        )
