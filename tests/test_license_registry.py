import json
from pathlib import Path

import pytest

from vyrex_ai.data.license_registry import (
    LicenseRecord,
    build_license_exclusions,
    load_license_registry,
)
from vyrex_ai.data.readers import Document


def valid_record(path: Path) -> dict:
    return {
        "source_path": str(path.resolve()),
        "source_name": "Test corpus",
        "domain": "general_english",
        "license_id": "CC0-1.0",
        "source_url": "https://example.org/dataset",
        "license_url": "https://example.org/license",
        "training_use": True,
        "commercial_use": True,
        "redistribution": True,
        "attribution_required": False,
        "share_alike": False,
        "review_status": "approved",
        "verified_at": "2026-11-14",
        "reviewer": "license-review",
    }


def test_approved_commercial_training_record_is_eligible(tmp_path):
    record = LicenseRecord.from_mapping(valid_record(tmp_path / "a.txt"))
    assert record.ineligibility_reason() is None


def test_unknown_and_unapproved_records_are_rejected(tmp_path):
    data = valid_record(tmp_path / "a.txt")
    data["review_status"] = "pending"

    record = LicenseRecord.from_mapping(data)
    assert record.ineligibility_reason() is not None


def test_commercial_permission_is_required(tmp_path):
    data = valid_record(tmp_path / "a.txt")
    data["commercial_use"] = False

    record = LicenseRecord.from_mapping(data)
    assert "Commercial use" in record.ineligibility_reason()


def test_registry_rejects_duplicate_paths(tmp_path):
    source = tmp_path / "a.txt"
    registry = tmp_path / "registry.jsonl"
    record = valid_record(source)

    registry.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate source path"):
        load_license_registry(registry)


def test_missing_license_record_is_quarantined(tmp_path):
    document = Document(str((tmp_path / "unknown.txt").resolve()), "sample")
    exclusions = build_license_exclusions([document], {})

    assert document.source_path in exclusions
    assert "No verified license" in exclusions[document.source_path]


def test_redistribution_permission_is_separate_from_training(tmp_path):
    data = valid_record(tmp_path / "a.txt")
    data["redistribution"] = False

    record = LicenseRecord.from_mapping(data)
    assert record.ineligibility_reason() is None


def test_registry_loads_valid_jsonl(tmp_path):
    source = tmp_path / "a.txt"
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps(valid_record(source)) + "\n",
        encoding="utf-8",
    )

    records = load_license_registry(registry)
    assert len(records) == 1


def test_missing_required_fields_are_rejected():
    with pytest.raises(ValueError, match="Missing license fields"):
        LicenseRecord.from_mapping({"source_path": "sample.txt"})



def test_pipeline_quarantines_unregistered_documents(tmp_path):
    from vyrex_ai.data.pipeline import prepare_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "unregistered.txt"
    source.write_text("English learning material. " * 10, encoding="utf-8")

    report = prepare_dataset(
        raw,
        source_groups={str(source.resolve()): "unknown"},
        license_registry_path=tmp_path / "empty.jsonl",
    )

    assert report.accepted_count == 0
    assert len(report.quarantined_documents) == 1
    assert "No verified license" in report.quarantined_documents[0].reason


def test_pipeline_accepts_verified_commercial_training_record(tmp_path):
    from vyrex_ai.data.pipeline import prepare_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "approved.txt"
    source.write_text("English learning material. " * 10, encoding="utf-8")

    registry = tmp_path / "licenses.jsonl"
    registry.write_text(
        json.dumps(valid_record(source)) + "\n",
        encoding="utf-8",
    )

    report = prepare_dataset(
        raw,
        source_groups={str(source.resolve()): "approved_source"},
        license_registry_path=registry,
        output_ratios=(1.0, 0.0, 0.0),
    )

    assert report.accepted_count == 1
    assert report.quarantined_documents == ()
    assert len(report.license_records) == 1


def test_manifest_preserves_license_evidence(tmp_path):
    from vyrex_ai.data.manifest import build_manifest
    from vyrex_ai.data.pipeline import prepare_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "approved.txt"
    source.write_text("English learning material. " * 10, encoding="utf-8")

    registry = tmp_path / "licenses.jsonl"
    registry.write_text(
        json.dumps(valid_record(source)) + "\n",
        encoding="utf-8",
    )

    groups = {str(source.resolve()): "approved_source"}
    report = prepare_dataset(
        raw,
        groups,
        license_registry_path=registry,
        output_ratios=(1.0, 0.0, 0.0),
    )
    entries = build_manifest(report, groups)

    assert len(entries) == 1
    assert entries[0].license_evidence["license_id"] == "CC0-1.0"
    assert entries[0].license_evidence["commercial_use"] is True
