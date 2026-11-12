import json
import sys
from pathlib import Path

import pytest

from vyrex_ai.cli import load_source_groups, main


def test_load_source_groups_resolves_relative_paths(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    source = raw / "chapter.txt"
    source.write_text("English learning material.", encoding="utf-8")

    mapping_file = tmp_path / "groups.json"
    mapping_file.write_text(
        json.dumps({"chapter.txt": "book_a"}),
        encoding="utf-8",
    )

    groups = load_source_groups(mapping_file, raw)

    assert groups == {str(source.resolve()): "book_a"}


def test_load_source_groups_rejects_paths_outside_input(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    mapping_file = tmp_path / "groups.json"
    mapping_file.write_text(
        json.dumps({"../outside.txt": "external"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the input directory"):
        load_source_groups(mapping_file, raw)


def test_prepare_data_cli_exports_dataset(tmp_path: Path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()

    source = raw / "chapter.txt"
    original = "English learning material for VYREX AI. " * 10
    source.write_text(original, encoding="utf-8")

    mapping_file = tmp_path / "groups.json"
    mapping_file.write_text(
        json.dumps({"chapter.txt": "book_a"}),
        encoding="utf-8",
    )

    output = tmp_path / "processed"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vyrex",
            "prepare-data",
            "--input", str(raw),
            "--output", str(output),
            "--groups", str(mapping_file),
            "--train-ratio", "1",
            "--validation-ratio", "0",
            "--test-ratio", "0",
        ],
    )

    main()

    captured = capsys.readouterr()

    assert "VYREX DATA PREPARATION COMPLETE" in captured.out
    assert "Training records: 1" in captured.out
    assert (output / "train.jsonl").exists()
    assert (output / "manifest.jsonl").exists()
    assert source.read_text(encoding="utf-8") == original


def test_train_tokenizer_cli_creates_artifacts_and_metadata(
    tmp_path: Path, monkeypatch, capsys
):
    dataset = tmp_path / "synthetic"
    dataset.mkdir()

    records = [
        {"text": "English learning begins with clear sentences."},
        {"text": "Reading and writing improve English skills."},
    ]
    train_file = dataset / "train.jsonl"
    train_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    output = tmp_path / "tokenizer"

    arguments = [
        "vyrex",
        "train-tokenizer",
        "--dataset", str(dataset),
        "--output", str(output),
        "--dataset-name", "synthetic-test",
        "--vocab-size", "300",
        "--min-frequency", "1",
    ]

    monkeypatch.setattr(sys, "argv", arguments)
    main()

    captured = capsys.readouterr()
    assert "VYREX TOKENIZER TRAINING COMPLETE" in captured.out
    assert "Training documents: 2" in captured.out

    tokenizer_file = output / "tokenizer.json"
    metadata_file = output / "tokenizer_metadata.json"

    assert tokenizer_file.is_file()
    assert metadata_file.is_file()

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["training_corpus"]["dataset_name"] == "synthetic-test"
    assert metadata["training_corpus"]["document_count"] == 2
    assert metadata["tokenizer"]["requested_vocab_size"] == 300

    with pytest.raises(FileExistsError, match="already exist"):
        main()

    monkeypatch.setattr(sys, "argv", arguments + ["--overwrite"])
    main()

    assert tokenizer_file.is_file()
    assert metadata_file.is_file()


def test_train_tokenizer_cli_rejects_empty_dataset_name(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vyrex",
            "train-tokenizer",
            "--dataset", str(tmp_path / "missing-dataset"),
            "--output", str(tmp_path / "tokenizer"),
            "--dataset-name", "   ",
        ],
    )

    with pytest.raises(ValueError, match="dataset-name cannot be empty"):
        main()
