import json

import pytest

from vyrex_ai.data.english_level_ml_expansion import (
    expand_tatoeba_candidates,
    normalize_text,
)


def make_record(sentence_id="101", text="The cat is sleeping on the sofa."):
    return {
        "sentence_id": sentence_id,
        "text": text,
        "language": "eng",
        "license": "CC0-1.0",
        "author": None,
        "source_archive": "eng_sentences_CC0.tsv.bz2",
        "source_sha256": "a" * 64,
        "base_sentence_id": "0",
        "quality_bucket": "accepted",
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_normalize_text():
    assert normalize_text("  English   LANGUAGE  ") == "english language"


def test_expansion_preserves_provisional_status(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "candidates.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(source, [make_record()])

    result = expand_tatoeba_candidates(source, output, manifest)

    candidate = json.loads(output.read_text(encoding="utf-8"))

    assert result["candidates_created"] == 1
    assert candidate["review_status"] == "pending"
    assert candidate["assigned_level"] is None
    assert candidate["prediction"]["confidence"] is None
    assert candidate["prediction"]["classification_status"] == (
        "provisional_not_human_approved"
    )
    assert result["human_approved_records"] == 0
    assert result["production_model_approved"] is False


def test_duplicate_normalized_text_is_excluded(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "candidates.jsonl"
    manifest = tmp_path / "manifest.jsonl"

    write_jsonl(source, [
        make_record("101", "The cat is sleeping on the sofa."),
        make_record("102", "  THE CAT IS SLEEPING ON THE SOFA.  "),
    ])

    result = expand_tatoeba_candidates(source, output, manifest)

    assert result["candidates_created"] == 1
    assert result["excluded"]["duplicate_normalized_text"] == 1


def test_duplicate_sentence_id_is_rejected(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [
        make_record("101"),
        make_record("101", "A different sentence."),
    ])

    with pytest.raises(ValueError, match="Duplicate sentence ID"):
        expand_tatoeba_candidates(
            source,
            tmp_path / "candidates.jsonl",
            tmp_path / "manifest.jsonl",
        )


def test_invalid_json_is_rejected(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        expand_tatoeba_candidates(
            source,
            tmp_path / "candidates.jsonl",
            tmp_path / "manifest.jsonl",
        )


def test_non_english_record_is_excluded(tmp_path):
    source = tmp_path / "input.jsonl"
    record = make_record()
    record["language"] = "fra"
    write_jsonl(source, [record])

    result = expand_tatoeba_candidates(
        source,
        tmp_path / "candidates.jsonl",
        tmp_path / "manifest.jsonl",
    )

    assert result["candidates_created"] == 0
    assert result["excluded"]["not_english"] == 1


def test_unsupported_license_is_excluded(tmp_path):
    source = tmp_path / "input.jsonl"
    record = make_record()
    record["license"] = "UNKNOWN"
    write_jsonl(source, [record])

    result = expand_tatoeba_candidates(
        source,
        tmp_path / "candidates.jsonl",
        tmp_path / "manifest.jsonl",
    )

    assert result["candidates_created"] == 0
    assert result["excluded"]["unsupported_license"] == 1


def test_cc_by_requires_attribution(tmp_path):
    source = tmp_path / "input.jsonl"
    record = make_record()
    record["license"] = "CC-BY-2.0-FR"
    write_jsonl(source, [record])

    with pytest.raises(ValueError, match="Missing CC-BY attribution"):
        expand_tatoeba_candidates(
            source,
            tmp_path / "candidates.jsonl",
            tmp_path / "manifest.jsonl",
        )


def test_cc_by_attribution_is_preserved(tmp_path):
    source = tmp_path / "input.jsonl"
    record = make_record()
    record["license"] = "CC-BY-2.0-FR"
    record["author"] = "Example Author"
    record["attribution"] = {
        "author": "Example Author",
        "source": "Tatoeba",
    }
    write_jsonl(source, [record])

    expand_tatoeba_candidates(
        source,
        tmp_path / "candidates.jsonl",
        tmp_path / "manifest.jsonl",
    )

    candidate = json.loads(
        (tmp_path / "candidates.jsonl").read_text(encoding="utf-8")
    )

    assert candidate["license"] == "CC-BY-2.0-FR"
    assert candidate["attribution"]["author"] == "Example Author"


def test_existing_output_is_never_overwritten(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "candidates.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(source, [make_record()])
    output.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        expand_tatoeba_candidates(source, output, manifest)

    assert output.read_text(encoding="utf-8") == "preserve me"


def test_candidate_id_is_deterministic(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [make_record()])

    first = expand_tatoeba_candidates(
        source,
        tmp_path / "first.jsonl",
        tmp_path / "first_manifest.jsonl",
    )
    second = expand_tatoeba_candidates(
        source,
        tmp_path / "second.jsonl",
        tmp_path / "second_manifest.jsonl",
    )

    first_record = json.loads(
        (tmp_path / "first.jsonl").read_text(encoding="utf-8")
    )
    second_record = json.loads(
        (tmp_path / "second.jsonl").read_text(encoding="utf-8")
    )

    assert first["candidates_created"] == second["candidates_created"] == 1
    assert first_record["candidate_id"] == second_record["candidate_id"]


def test_maximum_records_limits_input(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [
        make_record("101"),
        make_record("102", "A bird is flying above the trees."),
    ])

    result = expand_tatoeba_candidates(
        source,
        tmp_path / "candidates.jsonl",
        tmp_path / "manifest.jsonl",
        maximum_records=1,
    )

    assert result["records_examined"] == 1
    assert result["candidates_created"] == 1


def test_invalid_maximum_records_is_rejected(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [make_record()])

    with pytest.raises(ValueError, match="maximum_records"):
        expand_tatoeba_candidates(
            source,
            tmp_path / "candidates.jsonl",
            tmp_path / "manifest.jsonl",
            maximum_records=0,
        )


def test_dry_run_creates_no_output_files(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [make_record()])

    result = expand_tatoeba_candidates(source, dry_run=True)

    assert result["dry_run"] is True
    assert result["records_examined"] == 1
    assert result["candidates_created"] == 1
    assert result["source_modified"] is False
    assert not (tmp_path / "candidates.jsonl").exists()
    assert not (tmp_path / "manifest.jsonl").exists()


def test_license_counts_are_reported_separately(tmp_path):
    source = tmp_path / "input.jsonl"

    cc0 = make_record("101")
    cc_by = make_record("102", "A different sentence.")
    cc_by["license"] = "CC-BY-2.0-FR"
    cc_by["author"] = "Example Author"
    cc_by["attribution"] = {
        "author": "Example Author",
        "source": "Tatoeba",
    }

    write_jsonl(source, [cc0, cc_by])

    result = expand_tatoeba_candidates(
        source,
        tmp_path / "candidates.jsonl",
        tmp_path / "manifest.jsonl",
    )

    assert result["license_counts"] == {
        "CC-BY-2.0-FR": 1,
        "CC0-1.0": 1,
    }


def test_dry_run_and_normal_run_have_matching_statistics(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [
        make_record("101"),
        make_record("102", "A bird is flying above the trees."),
    ])

    dry_result = expand_tatoeba_candidates(source, dry_run=True)

    normal_result = expand_tatoeba_candidates(
        source,
        tmp_path / "candidates.jsonl",
        tmp_path / "manifest.jsonl",
    )

    for key in (
        "records_examined",
        "candidates_created",
        "excluded",
        "license_counts",
        "review_priority_counts",
        "provisional_level_counts",
    ):
        assert dry_result[key] == normal_result[key]
