import builtins
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vyrex_ai.cli import main, run_training, run_curriculum_training
from vyrex_ai.config import load_config


def _training_inputs(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    for split in ("train", "validation"):
        (dataset / f"{split}.jsonl").write_text(
            '{"text":"Synthetic English training material for testing."}\n',
            encoding="utf-8",
        )

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("placeholder", encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    )
    return dataset, tokenizer, checkpoint_dir, load_config(config_path)


def _training_args(dataset, tokenizer, checkpoint_dir, resume=None):
    return SimpleNamespace(
        dataset=str(dataset),
        tokenizer=str(tokenizer),
        checkpoint_dir=str(checkpoint_dir),
        device="cpu",
        resume=str(resume) if resume is not None else None,
        max_validation_batches=None,
    )


def test_train_rejects_unrecognized_resume_filename(tmp_path: Path):
    dataset, tokenizer, checkpoint_dir, config = _training_inputs(tmp_path)
    invalid_checkpoint = checkpoint_dir / "custom.pt"
    invalid_checkpoint.write_bytes(b"placeholder")
    args = _training_args(dataset, tokenizer, checkpoint_dir, invalid_checkpoint)

    with pytest.raises(ValueError, match="latest.pt or best.pt"):
        run_training(args, config, logging.getLogger("test"))


def test_train_rejects_checkpoint_directory_symlink(tmp_path: Path):
    target = tmp_path / "real-checkpoints"
    target.mkdir()
    link = tmp_path / "checkpoint-link"

    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    )
    config = load_config(config_path)
    args = _training_args(
        tmp_path / "missing-dataset",
        tmp_path / "missing-tokenizer.json",
        link,
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_training(args, config, logging.getLogger("test"))


def test_train_rejects_resume_checkpoint_symlink(tmp_path: Path):
    dataset, tokenizer, checkpoint_dir, config = _training_inputs(tmp_path)
    target = tmp_path / "external-checkpoint.pt"
    target.write_bytes(b"placeholder")
    link = checkpoint_dir / "latest.pt"

    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks unavailable: {exc}")

    args = _training_args(dataset, tokenizer, checkpoint_dir, link)

    with pytest.raises(ValueError, match="symbolic link"):
        run_training(args, config, logging.getLogger("test"))


def test_cli_help_does_not_import_training_dependencies(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "numpy"}:
            raise AssertionError(
                f"Optional training dependency imported: {name}"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["vyrex", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0

def test_train_rejects_symlink_in_checkpoint_parent(tmp_path: Path):
    target = tmp_path / "real-parent"
    target.mkdir()
    link = tmp_path / "linked-parent"

    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    )
    config = load_config(config_path)
    args = _training_args(
        tmp_path / "missing-dataset",
        tmp_path / "missing-tokenizer.json",
        link / "run1",
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_training(args, config, logging.getLogger("test"))

def test_train_rejects_other_checkpoint_symlink(tmp_path: Path):
    dataset, tokenizer, checkpoint_dir, config = _training_inputs(tmp_path)

    latest = checkpoint_dir / "latest.pt"
    latest.write_bytes(b"placeholder")

    external = tmp_path / "external-best.pt"
    external.write_bytes(b"placeholder")
    best_link = checkpoint_dir / "best.pt"

    try:
        best_link.symlink_to(external)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks unavailable: {exc}")

    args = _training_args(
        dataset,
        tokenizer,
        checkpoint_dir,
        resume=latest,
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_training(args, config, logging.getLogger("test"))


def _curriculum_config():
    from dataclasses import replace

    from vyrex_ai.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "base-v2.yaml")
    assert config.curriculum is not None
    return config


def _curriculum_args(tmp_path, resume=None, num_workers=0):
    dataset = tmp_path / "curriculum"
    dataset.mkdir(exist_ok=True)

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("placeholder", encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    return SimpleNamespace(
        dataset=str(dataset),
        tokenizer=str(tokenizer),
        checkpoint_dir=str(checkpoint_dir),
        device="cpu",
        resume=str(resume) if resume is not None else None,
        max_validation_batches=None,
        num_workers=num_workers,
    )


def test_curriculum_cli_rejects_config_without_curriculum(tmp_path):
    _, _, _, config = _training_inputs(tmp_path)
    args = _curriculum_args(tmp_path)

    with pytest.raises(ValueError, match="does not define a curriculum"):
        run_curriculum_training(
            args,
            config,
            logging.getLogger("test"),
        )


def test_curriculum_cli_requires_zero_workers(tmp_path):
    config = _curriculum_config()
    args = _curriculum_args(tmp_path, num_workers=2)

    with pytest.raises(ValueError, match="requires --num-workers 0"):
        run_curriculum_training(
            args,
            config,
            logging.getLogger("test"),
        )


def test_curriculum_cli_rejects_best_checkpoint_resume(tmp_path):
    config = _curriculum_config()
    args = _curriculum_args(tmp_path)

    best_checkpoint = Path(args.checkpoint_dir) / "best.pt"
    best_checkpoint.write_bytes(b"placeholder")
    args.resume = str(best_checkpoint)

    with pytest.raises(ValueError, match="latest.pt"):
        run_curriculum_training(
            args,
            config,
            logging.getLogger("test"),
        )


def test_curriculum_cli_help_does_not_import_training_dependencies(
    monkeypatch,
):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "numpy"}:
            raise AssertionError(
                f"Optional training dependency imported: {name}"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        sys,
        "argv",
        ["vyrex", "train-curriculum", "--help"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0

