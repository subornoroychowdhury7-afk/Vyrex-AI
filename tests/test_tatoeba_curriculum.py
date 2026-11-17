import json

import pytest

from vyrex_ai.data.tatoeba_curriculum import (
    TATOEBA_LEVELS,
    TatoebaAssignment,
    build_tatoeba_review_batch,
    load_tatoeba_assignments,
    validate_tatoeba_assignments,
    write_tatoeba_assignments,
)


HASH = "a" * 64


def make_record(sentence_id="101", text="The student reads a book."):
    return {
        "sentence_id": sentence_id,
        "text": text,
        "license": "CC0-1.0",
        "author": "test-author",
        "source_archive": "eng_sentences_CC0.tsv.bz2",
        "source_sha256": HASH,
        "quality_bucket": "accepted",
        "quality_reasons": [],
    }


def make_assignment(sentence_id="101", **overrides):
    values = {
        "sentence_id": sentence_id,
        "level": "basic_english",
        "rationale": "Reviewer judged vocabulary and sentence structure suitable.",
        "reviewer": "human-reviewer",
        "reviewed_at": "2026-11-13T10:00:00+05:30",
        "source_sha256": HASH,
        "license": "CC0-1.0",
    }
    values.update(overrides)
    return TatoebaAssignment(**values)


def test_review_batch_preserves_provenance_and_leaves_level_unassigned(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        json.dumps(make_record()) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "batch"
    summary = build_tatoeba_review_batch(source, output, limit=10)

    record = json.loads(
        (output / "review_queue.jsonl").read_text(encoding="utf-8")
    )

    assert summary["selected_records"] == 1
    assert record["sentence_id"] == "101"
    assert record["license"] == "CC0-1.0"
    assert record["author"] == "test-author"
    assert record["source_sha256"] == HASH
    assert record["review_status"] == "pending"
    assert record["assigned_level"] is None
    assert record["signals"]["token_count"] == 5
    assert source.read_text(encoding="utf-8").startswith('{"sentence_id"')


def test_review_batch_paginates_without_overlap(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        "".join(
            json.dumps(make_record(str(i))) + "\n"
            for i in range(1, 6)
        ),
        encoding="utf-8",
    )

    first = tmp_path / "batch1"
    second = tmp_path / "batch2"

    first_summary = build_tatoeba_review_batch(
        source, first, offset=0, limit=2
    )
    second_summary = build_tatoeba_review_batch(
        source, second, offset=2, limit=2
    )

    first_ids = [
        json.loads(line)["sentence_id"]
        for line in (first / "review_queue.jsonl").read_text().splitlines()
    ]
    second_ids = [
        json.loads(line)["sentence_id"]
        for line in (second / "review_queue.jsonl").read_text().splitlines()
    ]

    assert first_ids == ["1", "2"]
    assert second_ids == ["3", "4"]
    assert set(first_ids).isdisjoint(second_ids)
    assert first_summary["next_offset"] == 2
    assert second_summary["remaining_records"] == 1


def test_review_batch_refuses_to_overwrite(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(json.dumps(make_record()) + "\n", encoding="utf-8")
    output = tmp_path / "batch"
    build_tatoeba_review_batch(source, output)

    with pytest.raises(FileExistsError):
        build_tatoeba_review_batch(source, output)


def test_review_batch_rejects_invalid_arguments(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(json.dumps(make_record()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_tatoeba_review_batch(source, tmp_path / "bad1", offset=-1)

    with pytest.raises(ValueError):
        build_tatoeba_review_batch(source, tmp_path / "bad2", limit=0)


def test_human_assignment_round_trip(tmp_path):
    path = tmp_path / "assignments.json"
    assignment = make_assignment()

    write_tatoeba_assignments([assignment], path)
    loaded = load_tatoeba_assignments(path)

    assert loaded["101"] == assignment
    assert len(TATOEBA_LEVELS) == 4


def test_assignment_validation_rejects_duplicates_and_bad_levels():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_tatoeba_assignments([
            make_assignment(),
            make_assignment(),
        ])

    with pytest.raises(ValueError, match="Unsupported"):
        validate_tatoeba_assignments([
            make_assignment(level="english_aptitude")
        ])


def test_assignment_requires_timezone_aware_review_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        validate_tatoeba_assignments([
            make_assignment(reviewed_at="2026-11-13T10:00:00")
        ])


def test_assignment_requires_rationale_and_valid_hash():
    with pytest.raises(ValueError):
        validate_tatoeba_assignments([
            make_assignment(rationale=" ")
        ])

    with pytest.raises(ValueError):
        validate_tatoeba_assignments([
            make_assignment(source_sha256="not-a-hash")
        ])
