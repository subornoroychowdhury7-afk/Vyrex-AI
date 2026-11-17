import json

import pytest

from vyrex_ai.data.english_level_ml_data import (
    load_human_labelled_data,
)


def make_record(**overrides):
    record = {
        "sentence_id": "1001",
        "text": "The children are playing outside.",
        "review_status": "approved",
        "assigned_level": "basic_english",
        "rationale": "Simple vocabulary and sentence structure.",
        "reviewer": "Test Reviewer",
        "reviewed_at": "2026-11-13T10:00:00+05:30",
    }
    record.update(overrides)
    return record


def write_records(path, records, bom=False):
    content = "\n".join(json.dumps(r) for r in records) + "\n"
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text(content, encoding=encoding)


def test_loads_approved_human_label(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(path, [make_record()])

    result = load_human_labelled_data(path)

    assert result["eligible_records"] == 1
    assert result["examples"][0].label == "basic_english"
    assert len(result["examples"][0].features) == 20


def test_supports_utf8_bom(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(path, [make_record()], bom=True)

    result = load_human_labelled_data(path)

    assert result["eligible_records"] == 1


def test_excludes_unapproved_record(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(
        path,
        [make_record(review_status="pending")],
    )

    result = load_human_labelled_data(path)

    assert result["eligible_records"] == 0
    assert result["excluded"][0]["reason"] == "not human-approved"


def test_excludes_weak_rationale(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(path, [make_record(rationale="idk")])

    result = load_human_labelled_data(path)

    assert result["eligible_records"] == 0
    assert "rationale too short" in result["excluded"][0]["reason"]


def test_rejects_duplicate_sentence_ids(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(path, [make_record(), make_record()])

    with pytest.raises(ValueError, match="Duplicate sentence_id"):
        load_human_labelled_data(path)


def test_excludes_invalid_label(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(
        path,
        [make_record(assigned_level="unknown_level")],
    )

    result = load_human_labelled_data(path)

    assert result["eligible_records"] == 0


def test_excludes_invalid_timestamp(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(
        path,
        [make_record(reviewed_at="not-a-date")],
    )

    result = load_human_labelled_data(path)

    assert result["eligible_records"] == 0


def test_rejects_invalid_minimum_rationale_words(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(path, [make_record()])

    with pytest.raises(ValueError):
        load_human_labelled_data(path, minimum_rationale_words=0)


def test_preserves_source_file(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_records(path, [make_record()])
    original = path.read_bytes()

    load_human_labelled_data(path)

    assert path.read_bytes() == original
