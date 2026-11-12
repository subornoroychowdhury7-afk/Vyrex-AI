from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vyrex_ai.trainer import (
    _resolve_precision,
    train_model,
)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32)

    def forward(self, input_ids, targets=None):
        logits = self.head(self.embedding(input_ids))
        loss = None

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss


@pytest.fixture
def config():
    return SimpleNamespace(
        learning_rate=0.001,
        min_learning_rate=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        warmup_steps=0,
        max_steps=1,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        log_every_steps=0,
        eval_every_steps=1,
        checkpoint_every_steps=1,
    )


def test_fp32_precision_on_cpu():
    assert _resolve_precision(torch.device("cpu"), "fp32") is None


def test_invalid_precision_rejected():
    with pytest.raises(ValueError, match="precision"):
        _resolve_precision(torch.device("cpu"), "int8")


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_mixed_precision_rejected_on_cpu(precision):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        _resolve_precision(torch.device("cpu"), precision)


def test_fp32_training_completes_on_cpu(tmp_path, config):
    model = TinyLanguageModel()
    batch = torch.randint(0, 32, (2, 6))

    result = train_model(
        model=model,
        train_loader=[batch],
        validation_loader=[batch],
        training_config=config,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    assert result["completed_steps"] == 1
    assert result["precision"] == "fp32"
    assert result["device"] == "cpu"
    assert len(result["history"]) == 1
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()


def test_mixed_precision_training_rejected_before_training(
    tmp_path, config
):
    config.precision = "fp16"

    with pytest.raises(RuntimeError, match="requires CUDA"):
        train_model(
            model=TinyLanguageModel(),
            train_loader=[],
            validation_loader=[],
            training_config=config,
            checkpoint_dir=tmp_path,
            device="cpu",
        )
