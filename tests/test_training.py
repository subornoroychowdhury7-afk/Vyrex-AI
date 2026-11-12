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

    original_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

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

    metadata = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        map_location="cpu",
    )

    assert metadata["step"] == 25
    assert metadata["best_validation_loss"] == pytest.approx(1.234)

    for restored, original in zip(model.parameters(), original_parameters):
        assert torch.equal(restored, original)


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
