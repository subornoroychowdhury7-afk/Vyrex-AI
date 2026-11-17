import copy
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
        self.seen_batches = []

    def forward(self, input_ids, targets=None):
        self.seen_batches.append(input_ids.detach().cpu().clone())
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

def test_validation_loss_is_weighted_by_target_token_count():
    from vyrex_ai.trainer import _evaluate

    class FixedLossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.calls = 0

        def forward(self, input_ids, targets=None):
            self.calls += 1
            losses = [2.0, 8.0]
            loss = self.anchor + losses[self.calls - 1]
            return torch.empty(0), loss

    model = FixedLossModel()
    first_batch = torch.ones((2, 5), dtype=torch.long)
    second_batch = torch.ones((1, 3), dtype=torch.long)

    result = _evaluate(
        model,
        [first_batch, second_batch],
        torch.device("cpu"),
    )

    # First batch: 8 target tokens at loss 2.
    # Second batch: 2 target tokens at loss 8.
    assert result == pytest.approx((8 * 2.0 + 2 * 8.0) / 10)

def test_resume_continues_from_saved_training_batch(tmp_path, config):
    first_batch = torch.full((2, 6), 3, dtype=torch.long)
    second_batch = torch.full((2, 6), 7, dtype=torch.long)
    validation_batch = torch.full((2, 6), 5, dtype=torch.long)

    first_model = TinyLanguageModel()
    config.max_steps = 1
    config.eval_every_steps = 1
    config.checkpoint_every_steps = 1

    train_model(
        model=first_model,
        train_loader=[first_batch, second_batch],
        validation_loader=[validation_batch],
        training_config=config,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    assert torch.equal(
        first_model.seen_batches[0],
        first_batch[:, :-1],
    )
    assert torch.equal(
        first_model.seen_batches[0],
        first_batch[:, :-1],
    )

    assert (tmp_path / "best.pt").is_file()
    (tmp_path / "latest.pt").write_bytes(b"deliberately corrupted")

    resumed_model = TinyLanguageModel()
    config.max_steps = 2

    train_model(
        model=resumed_model,
        train_loader=[first_batch, second_batch],
        validation_loader=[validation_batch],
        training_config=config,
        checkpoint_dir=tmp_path,
        device="cpu",
        resume_from=tmp_path / "best.pt",
    )

    assert torch.equal(
        resumed_model.seen_batches[0],
        second_batch[:, :-1],
    )



def test_gradient_accumulation_matches_equivalent_large_batch(
    tmp_path, config
):
    torch.manual_seed(1234)

    initial_model = TinyLanguageModel()
    accumulated_model = copy.deepcopy(initial_model)
    large_batch_model = copy.deepcopy(initial_model)

    first_batch = torch.randint(0, 32, (2, 6))
    second_batch = torch.randint(0, 32, (2, 6))
    combined_batch = torch.cat([first_batch, second_batch], dim=0)

    accumulated_config = copy.deepcopy(config)
    accumulated_config.max_steps = 1
    accumulated_config.gradient_accumulation_steps = 2
    accumulated_config.eval_every_steps = 0
    accumulated_config.checkpoint_every_steps = 0

    large_batch_config = copy.deepcopy(accumulated_config)
    large_batch_config.gradient_accumulation_steps = 1

    train_model(
        model=accumulated_model,
        train_loader=[first_batch, second_batch],
        validation_loader=[combined_batch],
        training_config=accumulated_config,
        checkpoint_dir=tmp_path / "accumulated",
        device="cpu",
    )

    train_model(
        model=large_batch_model,
        train_loader=[combined_batch],
        validation_loader=[combined_batch],
        training_config=large_batch_config,
        checkpoint_dir=tmp_path / "large_batch",
        device="cpu",
    )

    for accumulated, large_batch in zip(
        accumulated_model.parameters(),
        large_batch_model.parameters(),
    ):
        assert torch.allclose(
            accumulated,
            large_batch,
            rtol=1e-5,
            atol=1e-7,
        )

def test_interrupted_resume_matches_uninterrupted_training(tmp_path, config):
    from vyrex_ai.training import load_checkpoint

    class DropoutLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 8)
            self.dropout = nn.Dropout(p=0.25)
            self.head = nn.Linear(8, 32)

        def forward(self, input_ids, targets=None):
            hidden = self.dropout(self.embedding(input_ids))
            logits = self.head(hidden)
            loss = None

            if targets is not None:
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

            return logits, loss

    class InterruptAfterTwoBatches:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            yield from self.batches[:2]
            raise KeyboardInterrupt("Simulated training interruption")

    def assert_nested_equal(left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                assert_nested_equal(left_item, right_item)
        else:
            assert left == right

    batches = [
        torch.full((2, 6), token, dtype=torch.long)
        for token in (3, 7, 11, 15)
    ]
    validation_batch = torch.full((2, 6), 5, dtype=torch.long)

    full_config = copy.deepcopy(config)
    full_config.max_steps = 4
    full_config.eval_every_steps = 1
    full_config.checkpoint_every_steps = 1
    full_config.log_every_steps = 0

    interrupted_config = copy.deepcopy(full_config)

    torch.manual_seed(20261112)
    uninterrupted_model = DropoutLanguageModel()

    torch.manual_seed(20261112)
    resumed_model = DropoutLanguageModel()

    torch.manual_seed(20261113)
    train_model(
        model=uninterrupted_model,
        train_loader=batches,
        validation_loader=[validation_batch],
        training_config=full_config,
        checkpoint_dir=tmp_path / "uninterrupted",
        device="cpu",
    )

    interrupted_dir = tmp_path / "interrupted"

    torch.manual_seed(20261113)
    with pytest.raises(KeyboardInterrupt, match="Simulated"):
        train_model(
            model=resumed_model,
            train_loader=InterruptAfterTwoBatches(batches),
            validation_loader=[validation_batch],
            training_config=interrupted_config,
            checkpoint_dir=interrupted_dir,
            device="cpu",
        )

    assert (interrupted_dir / "latest.pt").is_file()

    train_model(
        model=resumed_model,
        train_loader=batches,
        validation_loader=[validation_batch],
        training_config=interrupted_config,
        checkpoint_dir=interrupted_dir,
        device="cpu",
        resume_from=interrupted_dir / "latest.pt",
    )

    for name, expected in uninterrupted_model.state_dict().items():
        actual = resumed_model.state_dict()[name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    uninterrupted_optimizer = torch.optim.AdamW(
        uninterrupted_model.parameters(),
        lr=full_config.learning_rate,
        betas=tuple(full_config.betas),
        weight_decay=full_config.weight_decay,
    )
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=interrupted_config.learning_rate,
        betas=tuple(interrupted_config.betas),
        weight_decay=interrupted_config.weight_decay,
    )

    load_checkpoint(
        tmp_path / "uninterrupted" / "latest.pt",
        uninterrupted_model,
        uninterrupted_optimizer,
        map_location="cpu",
        restore_rng=False,
    )
    load_checkpoint(
        interrupted_dir / "latest.pt",
        resumed_model,
        resumed_optimizer,
        map_location="cpu",
        restore_rng=False,
    )

    assert_nested_equal(
        uninterrupted_optimizer.state_dict(),
        resumed_optimizer.state_dict(),
    )
