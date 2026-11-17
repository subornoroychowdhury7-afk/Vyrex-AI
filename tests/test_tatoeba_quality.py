import json

import pytest

from vyrex_ai.data.tatoeba_quality import (
    classify_record,
    filter_tatoeba,
)


@pytest.mark.parametrize(
    ("text", "expected", "reason"),
    [
        ("I love learning English every day.", "accepted", None),
        ("Good morning.", "review", "short_sentence_manual_review"),
        (
            "This is a longer sentence with enough words to exceed "
            "the normal thirty token training threshold by quite "
            "a substantial margin for this test and this additional phrase ensures the example contains more than thirty whitespace separated tokens overall.",
            "review",
            "long_sentence_manual_review",
        ),
        ("Contact me at test@example.com", "rejected", "contains_email"),
        ("Visit https://example.com now please", "rejected", "contains_url"),
        ("This is sooooooo boring and slow.", "rejected", "long_character_repetition"),
        ("", "rejected", "empty_or_invalid_text"),
    ],
)
def test_classify_record(text, expected, reason):
    bucket, reasons, _ = classify_record({"text": text})
    assert bucket == expected
    if reason:
        assert reason in reasons


def test_filter_preserves_provenance_and_reconciles_counts(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"

    cc0 = {
        "sentence_id": "1",
        "text": "I enjoy reading English books every evening.",
        "license": "CC0-1.0",
        "source_sha256": "abc",
    }
    cc_by = {
        "sentence_id": "2",
        "text": "Good evening.",
        "license": "CC-BY-2.0-FR",
        "attribution": {"author": "alice", "source": "Tatoeba"},
    }

    (source / "cc0_candidates.jsonl").write_text(
        json.dumps(cc0) + "\n", encoding="utf-8"
    )
    (source / "cc_by_candidates.jsonl").write_text(
        json.dumps(cc_by) + "\n", encoding="utf-8"
    )

    summary = filter_tatoeba(source, output)

    accepted = [
        json.loads(x) for x in
        (output / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    review = [
        json.loads(x) for x in
        (output / "review.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["counts"]["total"] == 2
    assert summary["counts"]["accepted"] == 1
    assert summary["counts"]["review"] == 1
    assert summary["counts"]["rejected"] == 0

    assert accepted[0]["license"] == "CC0-1.0"
    assert accepted[0]["source_sha256"] == "abc"
    assert review[0]["attribution"]["author"] == "alice"
    assert review[0]["license"] == "CC-BY-2.0-FR"


def test_filter_refuses_to_overwrite(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "cc0_candidates.jsonl").write_text("", encoding="utf-8")
    (source / "cc_by_candidates.jsonl").write_text("", encoding="utf-8")

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        filter_tatoeba(source, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


