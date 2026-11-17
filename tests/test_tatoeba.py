import bz2
import json

import pytest

from vyrex_ai.data.tatoeba import import_tatoeba


def _archive(path, rows):
    with bz2.open(path, "wt", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _fixture(root):
    source = root / "source"
    source.mkdir()

    _archive(source / "eng_sentences_base.tsv.bz2", [
        ("1", "0"), ("2", "0"), ("3", "0"), ("4", "9"),
        ("5", r"\N"), ("6", "0"), ("7", "0"),
    ])
    _archive(source / "eng_sentences_CC0.tsv.bz2", [
        ("1", "eng", "Hello world.", "2026-01-01"),
        ("2", "eng", "CC0 version.", "2026-01-01"),
        ("7", "eng", "No detailed record.", "2026-01-01"),
    ])
    _archive(source / "eng_sentences_detailed.tsv.bz2", [
        ("1", "eng", "Hello world.", "alice", "2025-01-01", "2026-01-01"),
        ("2", "eng", "Different text.", "bob", "2025-01-01", "2026-01-01"),
        ("3", "eng", "A licensed original.", "carol", "2025-01-01", "2026-01-01"),
        ("4", "eng", "A translation.", "dan", "2025-01-01", "2026-01-01"),
        ("5", "eng", "Unknown relationship.", "eve", "2025-01-01", "2026-01-01"),
        ("6", "eng", "No author.", r"\N", "2025-01-01", "2026-01-01"),
    ])
    return source


def _read(path):
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8"
    ).splitlines()]


def test_import_joins_cc0_metadata_and_separates_licenses(tmp_path):
    source = _fixture(tmp_path)
    output = tmp_path / "output"

    summary = import_tatoeba(source, output)

    cc0 = _read(output / "cc0_candidates.jsonl")
    by = _read(output / "cc_by_candidates.jsonl")
    quarantine = _read(output / "quarantine.jsonl")

    assert [r["sentence_id"] for r in cc0] == ["1", "7"]
    assert cc0[0]["author"] == "alice"
    assert cc0[0]["author_metadata_status"] == "joined"
    assert cc0[0]["date_added"] == "2025-01-01"
    assert cc0[1]["author_metadata_status"] == "missing_detailed_record"

    assert [r["sentence_id"] for r in by] == ["3"]
    assert by[0]["license"] == "CC-BY-2.0-FR"
    assert by[0]["attribution"]["author"] == "carol"

    reasons = {r["quarantine_reason"] for r in quarantine}
    assert "CC0_detailed_text_or_language_mismatch" in reasons
    assert "translation_license_review_required" in reasons
    assert "unknown_original_relationship" in reasons
    assert "missing_author_attribution" in reasons

    assert summary["schema_version"] == 2
    assert summary["counts"]["cc0_detailed_matches"] == 2
    assert summary["counts"]["cc0_missing_detailed"] == 1
    assert summary["counts"]["cc0_candidates"] == 2
    assert summary["counts"]["cc_by_candidates"] == 1


def test_import_refuses_to_overwrite_existing_output(tmp_path):
    source = _fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        import_tatoeba(source, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_duplicate_base_id_fails_without_publishing_output(tmp_path):
    source = _fixture(tmp_path)
    base = source / "eng_sentences_base.tsv.bz2"
    _archive(base, [("1", "0"), ("1", "0")])
    output = tmp_path / "should_not_exist"

    with pytest.raises(ValueError, match="Duplicate base"):
        import_tatoeba(source, output)

    assert not output.exists()
