import csv
import json

import pytest

from vyrex_ai.data.tatoeba_review_csv import (
    CSV_COLUMNS,
    export_review_queue_csv,
    import_review_decisions_csv,
)


SOURCE_HASH = "a" * 64


def make_queue(path):
    records = [
        {
            "sentence_id": "1001",
            "text": "The child is reading a book.",
            "license": "CC0-1.0",
            "author": "Example",
            "source_archive": "eng_sentences_CC0.tsv.bz2",
            "source_sha256": SOURCE_HASH,
        },
        {
            "sentence_id": "1002",
            "text": "The committee reached a unanimous decision.",
            "license": "CC-BY-2.0-FR",
            "author": "ReviewerExample",
            "source_archive": "eng_sentences_detailed.tsv.bz2",
            "source_sha256": "b" * 64,
        },
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    return records


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_review_queue_csv_preserves_source(tmp_path):
    queue = tmp_path / "review_queue.jsonl"
    output = tmp_path / "review.csv"
    make_queue(queue)
    original = queue.read_bytes()

    result = export_review_queue_csv(queue, output)

    rows = read_csv(output)
    assert result["records_exported"] == 2
    assert tuple(rows[0].keys()) == CSV_COLUMNS
    assert rows[0]["sentence_id"] == "1001"
    assert rows[0]["review_status"] == "pending"
    assert rows[1]["attribution_source"] == "Tatoeba"
    assert queue.read_bytes() == original


def test_import_only_approved_decisions(tmp_path):
    queue = tmp_path / "queue.jsonl"
    csv_path = tmp_path / "review.csv"
    ledger = tmp_path / "assignments.json"
    make_queue(queue)
    export_review_queue_csv(queue, csv_path)

    rows = read_csv(csv_path)
    rows[0].update({
        "review_status": "approved",
        "assigned_level": "basic_english",
        "rationale": "Common vocabulary and simple present tense.",
        "reviewer": "human-reviewer",
        "reviewed_at": "2026-11-13T01:00:00+05:30",
    })
    rows[1]["review_status"] = "needs_review"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    result = import_review_decisions_csv(csv_path, ledger)

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert result["approved_assignments"] == 1
    assert result["excluded_rows"] == 1
    assert len(payload["assignments"]) == 1
    assert payload["assignments"][0]["sentence_id"] == "1001"


def test_import_rejects_invalid_level(tmp_path):
    queue = tmp_path / "queue.jsonl"
    csv_path = tmp_path / "review.csv"
    make_queue(queue)
    export_review_queue_csv(queue, csv_path)

    rows = read_csv(csv_path)
    rows[0].update({
        "review_status": "approved",
        "assigned_level": "automatic_level",
        "rationale": "Test",
        "reviewer": "reviewer",
        "reviewed_at": "2026-11-13T01:00:00+05:30",
    })

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Invalid assigned level"):
        import_review_decisions_csv(csv_path, tmp_path / "ledger.json")


def test_import_rejects_invalid_source_hash(tmp_path):
    queue = tmp_path / "queue.jsonl"
    csv_path = tmp_path / "review.csv"
    make_queue(queue)
    export_review_queue_csv(queue, csv_path)

    rows = read_csv(csv_path)
    rows[0]["source_sha256"] = "invalid"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Invalid source hash"):
        import_review_decisions_csv(csv_path, tmp_path / "ledger.json")


def test_import_rejects_missing_cc_by_attribution(tmp_path):
    queue = tmp_path / "queue.jsonl"
    csv_path = tmp_path / "review.csv"
    make_queue(queue)
    export_review_queue_csv(queue, csv_path)

    rows = read_csv(csv_path)
    rows[1]["attribution_source"] = ""

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="CC-BY attribution"):
        import_review_decisions_csv(csv_path, tmp_path / "ledger.json")


def test_import_rejects_duplicate_sentence_ids(tmp_path):
    queue = tmp_path / "queue.jsonl"
    csv_path = tmp_path / "review.csv"
    make_queue(queue)
    export_review_queue_csv(queue, csv_path)

    rows = read_csv(csv_path)
    rows[1]["sentence_id"] = rows[0]["sentence_id"]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Duplicate sentence ID"):
        import_review_decisions_csv(csv_path, tmp_path / "ledger.json")


def test_export_refuses_overwrite(tmp_path):
    queue = tmp_path / "queue.jsonl"
    output = tmp_path / "review.csv"
    make_queue(queue)
    output.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_review_queue_csv(queue, output)

    assert output.read_text(encoding="utf-8") == "preserve me"


def test_import_refuses_overwrite(tmp_path):
    queue = tmp_path / "queue.jsonl"
    csv_path = tmp_path / "review.csv"
    ledger = tmp_path / "ledger.json"
    make_queue(queue)
    export_review_queue_csv(queue, csv_path)
    ledger.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        import_review_decisions_csv(csv_path, ledger)

    assert ledger.read_text(encoding="utf-8") == "preserve me"


def test_import_rejects_no_approved_rows(tmp_path):
    queue = tmp_path / "queue.jsonl"
    csv_path = tmp_path / "review.csv"
    make_queue(queue)
    export_review_queue_csv(queue, csv_path)

    with pytest.raises(ValueError, match="No approved"):
        import_review_decisions_csv(csv_path, tmp_path / "ledger.json")
