from pathlib import Path
from vyrex_ai import curriculum_trainer as curriculum_trainer_module
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vyrex_ai.config import EXPECTED_CURRICULUM_STAGES
from vyrex_ai.curriculum_trainer import (
    _stage_step_counts,
    train_curriculum,
)


class TinyCurriculumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)

    def forward(self, inputs, targets=None):
        logits = self.head(self.embedding(inputs))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 8),
                targets.reshape(-1),
            )
        return logits, loss


def make_config(max_steps=5):
    return SimpleNamespace(
        learning_rate=0.001,
        min_learning_rate=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        warmup_steps=0,
        gradient_clip_norm=1.0,
        precision="fp32",
        log_every_steps=0,
        eval_every_steps=0,
        checkpoint_every_steps=1,
    )


def make_loaders():
    batch = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ], dtype=torch.long)

    return {
        stage: ([batch.clone()], [batch.clone()])
        for stage in EXPECTED_CURRICULUM_STAGES
    }


def test_stage_step_counts_distribute_remainder():
    assert _stage_step_counts(10000) == (2000, 2000, 2000, 2000, 2000)
    assert _stage_step_counts(10003) == (2001, 2001, 2001, 2000, 2000)


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_stage_step_counts_reject_too_few_steps(steps):
    with pytest.raises(ValueError):
        _stage_step_counts(steps)


def test_curriculum_trains_all_five_stages_with_one_optimizer(tmp_path):
    model = TinyCurriculumModel()

    result = train_curriculum(
        model=model,
        stage_loaders=make_loaders(),
        training_config=make_config(),
        checkpoint_dir=tmp_path / "checkpoints",
        resume_signature={"synthetic_dataset": "test-v1"},
        device="cpu",
    )

    assert result["completed_steps"] == 5
    assert result["completed_stages"] == list(EXPECTED_CURRICULUM_STAGES)
    assert len(result["history"]) == 5
    assert result["checkpoint"].endswith("latest.pt")

    checkpoint = torch.load(
        tmp_path / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint["curriculum_state"]
    assert state["global_step"] == 5
    assert state["active_stage_index"] == 5
    assert state["stage_step"] == 0
    assert state["completed_stages"] == list(EXPECTED_CURRICULUM_STAGES)

    # A single AdamW optimizer was retained across all five stages.
    optimizer_state = checkpoint["optimizer_state_dict"]["state"]
    assert optimizer_state
    assert all(
        int(parameter_state["step"].item()) == 5
        for parameter_state in optimizer_state.values()
    )


def test_curriculum_rejects_stage_order_mismatch(tmp_path):
    loaders = make_loaders()
    reversed_loaders = dict(reversed(list(loaders.items())))

    with pytest.raises(ValueError, match="approved stages in order"):
        train_curriculum(
            model=TinyCurriculumModel(),
            stage_loaders=reversed_loaders,
            training_config=make_config(),
            checkpoint_dir=tmp_path / "checkpoints",
            resume_signature={"synthetic_dataset": "test-v1"},
            device="cpu",
        )


def test_completed_curriculum_can_be_resumed_without_retraining(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    signature = {"synthetic_dataset": "test-v1"}

    first = train_curriculum(
        model=TinyCurriculumModel(),
        stage_loaders=make_loaders(),
        training_config=make_config(),
        checkpoint_dir=checkpoint_dir,
        resume_signature=signature,
        device="cpu",
    )

    second = train_curriculum(
        model=TinyCurriculumModel(),
        stage_loaders=make_loaders(),
        training_config=make_config(),
        checkpoint_dir=checkpoint_dir,
        resume_signature=signature,
        resume_from=checkpoint_dir / "latest.pt",
        device="cpu",
    )

    assert first["completed_steps"] == second["completed_steps"] == 5
    assert second["completed_stages"] == list(EXPECTED_CURRICULUM_STAGES)


def test_curriculum_rejects_resume_signature_mismatch(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"

    train_curriculum(
        model=TinyCurriculumModel(),
        stage_loaders=make_loaders(),
        training_config=make_config(),
        checkpoint_dir=checkpoint_dir,
        resume_signature={"synthetic_dataset": "test-v1"},
        device="cpu",
    )

    with pytest.raises(ValueError, match="signature mismatch"):
        train_curriculum(
            model=TinyCurriculumModel(),
            stage_loaders=make_loaders(),
            training_config=make_config(),
            checkpoint_dir=checkpoint_dir,
            resume_signature={"synthetic_dataset": "changed-v2"},
            resume_from=checkpoint_dir / "latest.pt",
            device="cpu",
        )

class TinyDropoutCurriculumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(8, 8)

    def forward(self, inputs, targets=None):
        logits = self.head(self.dropout(self.embedding(inputs)))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 8),
                targets.reshape(-1),
            )
        return logits, loss


class InterruptAfterOneBatch:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch.clone()
        raise KeyboardInterrupt("Synthetic interruption for recovery test")


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    signature = {"synthetic_dataset": "deterministic-v1"}
    reference_dir = tmp_path / "reference"
    resumed_dir = tmp_path / "resumed"

    reference_config = make_config(max_steps=10)
    reference_config.eval_every_steps = 1

    torch.manual_seed(12345)
    reference_model = TinyDropoutCurriculumModel()

    train_curriculum(
        model=reference_model,
        stage_loaders=make_loaders(),
        training_config=reference_config,
        checkpoint_dir=reference_dir,
        resume_signature=signature,
        device="cpu",
    )

    reference_checkpoint = torch.load(
        reference_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )

    interrupted_config = make_config(max_steps=10)
    interrupted_config.eval_every_steps = 1

    torch.manual_seed(12345)
    interrupted_model = TinyDropoutCurriculumModel()

    interrupted_loaders = make_loaders()
    first_stage = EXPECTED_CURRICULUM_STAGES[0]
    batch = torch.tensor(
        [[1, 2, 3, 4], [2, 3, 4, 5]],
        dtype=torch.long,
    )
    interrupted_loaders[first_stage] = (
        InterruptAfterOneBatch(batch),
        interrupted_loaders[first_stage][1],
    )

    with pytest.raises(KeyboardInterrupt, match="Synthetic interruption"):
        train_curriculum(
            model=interrupted_model,
            stage_loaders=interrupted_loaders,
            training_config=interrupted_config,
            checkpoint_dir=resumed_dir,
            resume_signature=signature,
            device="cpu",
        )

    saved = torch.load(
        resumed_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["curriculum_state"]["global_step"] == 1
    assert saved["curriculum_state"]["stage_step"] == 1

    train_curriculum(
        model=interrupted_model,
        stage_loaders=make_loaders(),
        training_config=interrupted_config,
        checkpoint_dir=resumed_dir,
        resume_signature=signature,
        resume_from=resumed_dir / "latest.pt",
        device="cpu",
    )

    resumed_checkpoint = torch.load(
        resumed_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert_nested_equal(
        reference_checkpoint["model_state_dict"],
        resumed_checkpoint["model_state_dict"],
    )
    assert_nested_equal(
        reference_checkpoint["optimizer_state_dict"],
        resumed_checkpoint["optimizer_state_dict"],
    )

def test_resume_recovers_after_final_stage_validation_before_transition(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "boundary-recovery"
    config = make_config(max_steps=5)
    signature = {"synthetic_dataset": "boundary-v1"}

    torch.manual_seed(2468)
    model = TinyCurriculumModel()
    original_save = curriculum_trainer_module._save_curriculum_checkpoint

    def save_then_interrupt(path, model, optimizer, state, scaler):
        original_save(path, model, optimizer, state, scaler)
        if (
            Path(path).name == "latest.pt"
            and state["active_stage_index"] == 0
            and state["stage_step"] == 1
        ):
            raise KeyboardInterrupt("Crash after final-step checkpoint")

    monkeypatch.setattr(
        curriculum_trainer_module,
        "_save_curriculum_checkpoint",
        save_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="final-step checkpoint"):
        train_curriculum(
            model=model,
            stage_loaders=make_loaders(),
            training_config=config,
            checkpoint_dir=checkpoint_dir,
            resume_signature=signature,
            device="cpu",
        )

    saved = torch.load(
        checkpoint_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    saved_state = saved["curriculum_state"]
    assert saved_state["active_stage_index"] == 0
    assert saved_state["stage_step"] == 1
    assert saved_state["global_step"] == 1
    assert saved_state["completed_stages"] == []

    monkeypatch.setattr(
        curriculum_trainer_module,
        "_save_curriculum_checkpoint",
        original_save,
    )

    result = train_curriculum(
        model=model,
        stage_loaders=make_loaders(),
        training_config=config,
        checkpoint_dir=checkpoint_dir,
        resume_signature=signature,
        resume_from=checkpoint_dir / "latest.pt",
        device="cpu",
    )

    assert result["completed_steps"] == 5
    assert result["completed_stages"] == list(EXPECTED_CURRICULUM_STAGES)

    final_checkpoint = torch.load(
        checkpoint_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    final_state = final_checkpoint["curriculum_state"]
    assert final_state["global_step"] == 5
    assert final_state["active_stage_index"] == 5
    assert final_state["stage_step"] == 0
