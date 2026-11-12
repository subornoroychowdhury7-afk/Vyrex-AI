from pathlib import Path

from vyrex_ai.data.pipeline import prepare_dataset


def test_pipeline_processes_documents_safely(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    text_a = "This is a useful English learning document. " * 10
    text_b = "This document explains grammar and vocabulary. " * 10
    text_c = "This document teaches reading comprehension. " * 10

    (input_dir / "a.txt").write_text(text_a, encoding="utf-8")
    (input_dir / "b.txt").write_text(text_b, encoding="utf-8")
    (input_dir / "c.txt").write_text(text_c, encoding="utf-8")
    (input_dir / "duplicate.txt").write_text(text_a, encoding="utf-8")
    (input_dir / "empty.txt").write_text("", encoding="utf-8")

    groups = {
        str((input_dir / "a.txt").resolve()): "source_a",
        str((input_dir / "b.txt").resolve()): "source_b",
        str((input_dir / "c.txt").resolve()): "source_c",
        str((input_dir / "duplicate.txt").resolve()): "source_duplicate",
    }

    report = prepare_dataset(
        input_dir,
        source_groups=groups,
        output_ratios=(0.5, 0.25, 0.25),
        min_chars=100,
    )

    assert report.total_read == 5
    assert report.accepted_count == 4
    assert report.rejected_count == 1
    assert report.duplicate_count == 1

    all_split_documents = (
        report.splits.train
        + report.splits.validation
        + report.splits.test
    )

    assert len(all_split_documents) == 3
    assert all("empty.txt" not in doc.source_path for doc in all_split_documents)


def test_pipeline_reports_read_failures(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    (input_dir / "broken.txt").write_bytes(b"\xff\xfe\xfa")

    report = prepare_dataset(
        input_dir,
        source_groups={},
        output_ratios=(1.0, 0.0, 0.0),
    )

    assert report.total_read == 0
    assert len(report.read_failures) == 1
    assert "UnicodeDecodeError" in report.read_failures[0].reason
    assert report.splits.train == ()


def test_pipeline_does_not_modify_original_files(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    original_text = "Preserve this original document. " * 10
    file_path = input_dir / "original.txt"
    file_path.write_text(original_text, encoding="utf-8")

    prepare_dataset(
        input_dir,
        source_groups={str(file_path.resolve()): "original_source"},
        output_ratios=(1.0, 0.0, 0.0),
    )

    assert file_path.read_text(encoding="utf-8") == original_text

def test_pipeline_handles_all_documents_rejected(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    (input_dir / "empty.txt").write_text("", encoding="utf-8")

    report = prepare_dataset(
        input_dir,
        source_groups={},
        output_ratios=(0.98, 0.01, 0.01),
    )

    assert report.total_read == 1
    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert len(report.splits.train) == 0
    assert len(report.splits.validation) == 0
    assert len(report.splits.test) == 0
