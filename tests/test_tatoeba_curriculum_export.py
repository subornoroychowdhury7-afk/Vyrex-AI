import json

import pytest

from vyrex_ai.data.tatoeba_curriculum import (
    TatoebaAssignment,
    write_tatoeba_assignments,
)
from vyrex_ai.data.tatoeba_curriculum_export import (
    export_reviewed_tatoeba,
)


SOURCE_HASH = "a" * 64


def record(sentence_id, text, license_name="CC0-1.0"):
    item = {
        "sentence_id": sentence_id,
        "text": text,
        "license": license_name,
        "source_sha256": SOURCE_HASH,
        "source_archive": "test.tsv.bz2",
        "quality_bucket": "accepted",
        "author": None,
    }
    if license_name == "CC-BY-2.0-FR":
        item["author"] = "test-author"
        item["attribution"] = {
            "author": "test-author",
            "source": "Tatoeba",
        }
    return item


def assignment(sentence_id, level="basic_english", **overrides):
    values = {
        "sentence_id": sentence_id,
        "level": level,
        "rationale": "Human reviewer assessed vocabulary and structure.",
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-11-13T10:00:00+05:30",
        "source_sha256": SOURCE_HASH,
        "license": "CC0-1.0",
    }
    values.update(overrides)
    return TatoebaAssignment(**values)


def write_source(path, records):
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def test_export_creates_four_cumulative_english_stages(tmp_path):
    source = tmp_path / "accepted.jsonl"
    records = [
        record("1", "The child reads a book."),
        record("2", "The student is learning English."),
        record("3", "Although it was raining, we continued."),
        record("4", "Notwithstanding the objections, the proposal prevailed."),
    ]
    write_source(source, records)

    manifest = tmp_path / "assignments.json"
    write_tatoeba_assignments([
        assignment("1", "basic_english"),
        assignment("2", "elementary_intermediate_english"),
        assignment("3", "upper_intermediate_english"),
        assignment("4", "advanced_english"),
    ], manifest)

    output = tmp_path / "export"
    summary = export_reviewed_tatoeba(
        source, manifest, output,
        train_ratio=1.0,
        validation_ratio=0.0,
        test_ratio=0.0,
    )

    assert summary["reviewed_sentence_count"] == 4
    assert summary["cross_split_duplicate_group_leaks"] == 0
    assert summary["stage_counts"]["basic_english"]["train"] == 1
    assert summary["stage_counts"]["elementary_intermediate_english"]["train"] == 2
    assert summary["stage_counts"]["upper_intermediate_english"]["train"] == 3
    assert summary["stage_counts"]["advanced_english"]["train"] == 4
    assert summary["aptitude_stage"]["status"] == (
        "blocked_pending_original_aptitude_dataset"
    )

    stage1 = [
        json.loads(line)
        for line in (output / "basic_english" / "train.jsonl").read_text().splitlines()
    ]
    assert stage1 == [{"text": "The child reads a book."}]


def test_exact_duplicate_text_never_crosses_splits(tmp_path):
    source = tmp_path / "accepted.jsonl"
    records = [
        record("1", "The same sentence appears here."),
        record("2", "  THE SAME SENTENCE   APPEARS HERE. "),
        record("3", "A different sentence appears here."),
        record("4", "Another sentence appears over there."),
        record("5", "Yet another sentence is present."),
    ]
    write_source(source, records)

    manifest = tmp_path / "assignments.json"
    write_tatoeba_assignments([
        assignment(str(i)) for i in range(1, 6)
    ], manifest)

    output = tmp_path / "export"
    summary = export_reviewed_tatoeba(
        source, manifest, output,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    audit = [
        json.loads(line)
        for line in (
            output / "basic_english" / "curriculum_manifest.jsonl"
        ).read_text().splitlines()
    ]
    split_by_id = {item["sentence_id"]: item["split"] for item in audit}

    assert split_by_id["1"] == split_by_id["2"]
    assert summary["cross_split_duplicate_group_leaks"] == 0


def test_export_rejects_source_hash_mismatch(tmp_path):
    source = tmp_path / "accepted.jsonl"
    write_source(source, [record("1", "A valid sentence is here.")])

    manifest = tmp_path / "assignments.json"
    write_tatoeba_assignments([
        assignment("1", source_sha256="b" * 64)
    ], manifest)

    with pytest.raises(ValueError, match="Source hash mismatch"):
        export_reviewed_tatoeba(source, manifest, tmp_path / "export")


def test_export_rejects_unknown_assigned_sentence(tmp_path):
    source = tmp_path / "accepted.jsonl"
    write_source(source, [record("1", "A valid sentence is here.")])

    manifest = tmp_path / "assignments.json"
    write_tatoeba_assignments([assignment("999")], manifest)

    with pytest.raises(ValueError, match="not found"):
        export_reviewed_tatoeba(source, manifest, tmp_path / "export")


def test_export_preserves_and_validates_cc_by_attribution(tmp_path):
    source = tmp_path / "accepted.jsonl"
    write_source(source, [
        record("1", "This attributed sentence is suitable.", "CC-BY-2.0-FR")
    ])

    manifest = tmp_path / "assignments.json"
    write_tatoeba_assignments([
        assignment("1", license="CC-BY-2.0-FR")
    ], manifest)

    output = tmp_path / "export"
    export_reviewed_tatoeba(
        source, manifest, output,
        train_ratio=1.0,
        validation_ratio=0.0,
        test_ratio=0.0,
    )

    audit = json.loads(
        (output / "basic_english" / "curriculum_manifest.jsonl")
        .read_text(encoding="utf-8").strip()
    )
    assert audit["license"] == "CC-BY-2.0-FR"
    assert audit["author"] == "test-author"
    assert audit["attribution"]["source"] == "Tatoeba"


def test_export_refuses_to_overwrite_existing_output(tmp_path):
    source = tmp_path / "accepted.jsonl"
    write_source(source, [record("1", "A valid sentence is here.")])

    manifest = tmp_path / "assignments.json"
    write_tatoeba_assignments([assignment("1")], manifest)

    output = tmp_path / "export"
    output.mkdir()

    with pytest.raises(FileExistsError):
        export_reviewed_tatoeba(source, manifest, output)


def test_export_rejects_non_accepted_record(tmp_path):
    source = tmp_path / "accepted.jsonl"
    item = record("1", "A valid sentence is here.")
    item["quality_bucket"] = "review"
    write_source(source, [item])

    manifest = tmp_path / "assignments.json"
    write_tatoeba_assignments([assignment("1")], manifest)

    with pytest.raises(ValueError, match="not quality-approved"):
        export_reviewed_tatoeba(source, manifest, tmp_path / "export")
