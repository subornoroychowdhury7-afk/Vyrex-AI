import json
from pathlib import Path

from vyrex_ai.data.manifest import build_manifest, write_manifest
from vyrex_ai.data.pipeline import prepare_dataset


def test_manifest_records_all_document_statuses(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    text = "Useful English learning material. " * 10

    (input_dir / "a.txt").write_text(text, encoding="utf-8")
    (input_dir / "b.txt").write_text(text, encoding="utf-8")
    (input_dir / "empty.txt").write_text("", encoding="utf-8")
    (input_dir / "broken.txt").write_bytes(b"\xff\xfe\xfa")

    groups = {
        str((input_dir / "a.txt").resolve()): "source_a",
        str((input_dir / "b.txt").resolve()): "source_b",
    }

    report = prepare_dataset(input_dir, groups, (1.0, 0.0, 0.0))
    entries = build_manifest(report, groups)

    assert {entry.status for entry in entries} == {
        "accepted", "duplicate", "rejected", "read_error"
    }
    assert len(entries) == 4


def test_manifest_writes_valid_jsonl(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    file_path = input_dir / "sample.txt"
    file_path.write_text("English learning material. " * 10, encoding="utf-8")

    groups = {str(file_path.resolve()): "sample_source"}
    report = prepare_dataset(input_dir, groups, (1.0, 0.0, 0.0))
    entries = build_manifest(report, groups)

    output = tmp_path / "manifest.jsonl"
    write_manifest(entries, output)

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 1
    assert records[0]["status"] == "accepted"
    assert records[0]["split"] == "train"


def test_manifest_preserves_original_document(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    original = "Preserve this original document. " * 10
    file_path = input_dir / "original.txt"
    file_path.write_text(original, encoding="utf-8")

    groups = {str(file_path.resolve()): "original_source"}
    report = prepare_dataset(input_dir, groups, (1.0, 0.0, 0.0))

    entries = build_manifest(report, groups)
    write_manifest(entries, tmp_path / "manifest.jsonl")

    assert file_path.read_text(encoding="utf-8") == original
