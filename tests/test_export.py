import json
from pathlib import Path

import pytest

from vyrex_ai.data.export import export_dataset
from vyrex_ai.data.pipeline import prepare_dataset


def test_export_writes_all_splits_and_manifest(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    groups = {}

    for index in range(10):
        path = raw / f"source_{index}.txt"
        path.write_text(
            f"Unique English learning document number {index}. " * 10,
            encoding="utf-8",
        )
        groups[str(path.resolve())] = f"group_{index}"

    report = prepare_dataset(
        raw,
        groups,
        output_ratios=(0.8, 0.1, 0.1),
    )

    output = tmp_path / "processed"
    result = export_dataset(report, groups, output)

    assert result.train_path.exists()
    assert result.validation_path.exists()
    assert result.test_path.exists()
    assert result.manifest_path.exists()

    assert result.train_count + result.validation_count + result.test_count == 10


def test_export_writes_valid_jsonl(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    source = raw / "sample.txt"
    original = "English learning material for VYREX. " * 10
    source.write_text(original, encoding="utf-8")

    groups = {str(source.resolve()): "sample_group"}
    report = prepare_dataset(raw, groups, (1.0, 0.0, 0.0))

    result = export_dataset(report, groups, tmp_path / "processed")

    records = [
        json.loads(line)
        for line in result.train_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 1
    assert records[0] == {"text": original}


def test_export_does_not_modify_original_files(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    source = raw / "original.txt"
    original = "Preserve this original English document. " * 10
    source.write_text(original, encoding="utf-8")

    groups = {str(source.resolve()): "original_group"}
    report = prepare_dataset(raw, groups, (1.0, 0.0, 0.0))

    export_dataset(report, groups, tmp_path / "processed")

    assert source.read_text(encoding="utf-8") == original


def test_export_rejects_document_appearing_in_multiple_splits(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    source = raw / "sample.txt"
    source.write_text("Valid English document. " * 10, encoding="utf-8")

    groups = {str(source.resolve()): "sample_group"}
    report = prepare_dataset(raw, groups, (1.0, 0.0, 0.0))

    duplicate_report = type(report)(
        total_read=report.total_read,
        accepted_count=report.accepted_count,
        rejected_count=report.rejected_count,
        duplicate_count=report.duplicate_count,
        read_failures=report.read_failures,
        validation_results=report.validation_results,
        deduplication=report.deduplication,
        splits=type(report.splits)(
            train=report.splits.train,
            validation=report.splits.train,
            test=(),
        ),
    )

    with pytest.raises(ValueError, match="more than one split"):
        export_dataset(duplicate_report, groups, tmp_path / "processed")


def test_export_includes_oanc_metadata_in_manifest(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    source = raw / "sample.txt"
    source.write_text("English learning material. " * 10, encoding="utf-8")

    metadata = raw / "sample.anc"
    metadata.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <cesHeader xmlns="http://www.xces.org/ns/GrAF/1.0/">
            <fileDesc>
                <titleStmt><title>Sample Document</title></titleStmt>
                <extent wordCount="40"/>
                <sourceDesc>
                    <title>Sample Corpus</title>
                    <author>Test Author</author>
                    <publisher>Test Publisher</publisher>
                </sourceDesc>
            </fileDesc>
            <profileDesc>
                <textClass catRef="WR JO"/>
            </profileDesc>
        </cesHeader>
        """,
        encoding="utf-8",
    )

    groups = {str(source.resolve()): "sample_group"}
    report = prepare_dataset(raw, groups, (1.0, 0.0, 0.0))

    result = export_dataset(report, groups, tmp_path / "processed")

    records = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 1
    assert records[0]["metadata"]["title"] == "Sample Document"
    assert records[0]["metadata"]["authors"] == ["Test Author"]
    assert records[0]["metadata"]["word_count"] == 40
    assert records[0]["metadata"]["category_codes"] == ["WR", "JO"]
    assert records[0]["metadata_error"] is None
    assert "raw_xml" not in records[0]["metadata"]


def test_oanc_metadata_never_enters_training_jsonl(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()

    source = raw / "sample.txt"
    original = "Clean English training text. " * 10
    source.write_text(original, encoding="utf-8")

    (raw / "sample.anc").write_text(
        """<cesHeader>
            <fileDesc>
                <titleStmt><title>Metadata Only</title></titleStmt>
            </fileDesc>
        </cesHeader>""",
        encoding="utf-8",
    )

    groups = {str(source.resolve()): "sample_group"}
    report = prepare_dataset(raw, groups, (1.0, 0.0, 0.0))

    result = export_dataset(report, groups, tmp_path / "processed")

    records = [
        json.loads(line)
        for line in result.train_path.read_text(encoding="utf-8").splitlines()
    ]

    assert records == [{"text": original}]
    assert "Metadata Only" not in result.train_path.read_text(encoding="utf-8")


def test_export_includes_oanc_document_classification(tmp_path: Path):
    raw = tmp_path / "OANC-GrAF"
    source = (
        raw / "data" / "spoken" / "telephone"
        / "switchboard" / "sample.txt"
    )
    source.parent.mkdir(parents=True)
    source.write_text("Natural spoken English conversation. " * 10,
                      encoding="utf-8")

    metadata = source.with_suffix(".anc")
    metadata.write_text(
        """<cesHeader>
            <fileDesc>
                <titleStmt><title>Telephone Sample</title></titleStmt>
            </fileDesc>
        </cesHeader>""",
        encoding="utf-8",
    )

    groups = {str(source.resolve()): "switchboard"}
    report = prepare_dataset(raw, groups, (1.0, 0.0, 0.0))

    result = export_dataset(
        report,
        groups,
        tmp_path / "processed",
        corpus_root=raw,
    )

    records = [
        json.loads(line)
        for line in result.manifest_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 1
    assert records[0]["metadata"]["document_type"] == "spoken"
    assert records[0]["metadata"]["collection"] == "switchboard"
    assert records[0]["metadata"]["genre"] == "telephone"

