import json
from pathlib import Path

import pytest

from vyrex_ai.data import english_level_v2_2_audit as audit


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def fake_features(text: str) -> dict:
    value = float(len(text))
    return {
        "lexical": {
            "word_count": value,
            "unique_word_count": value,
        },
        "syntactic": {
            "subordinate_marker_count": value,
        },
        "grammatical": {
            "modal_auxiliary_count": 0.0,
        },
        "frequency": {
            "frequency_coverage": 1.0,
        },
    }


def test_reservoir_sampling_is_deterministic(tmp_path):
    path = tmp_path / "input.jsonl"
    write_jsonl(
        path,
        [{"sentence_id": str(i), "text": f"sentence {i}"} for i in range(100)],
    )

    first, metadata1 = audit._reservoir_sample(path, 10, 42)
    second, metadata2 = audit._reservoir_sample(path, 10, 42)

    assert first == second
    assert metadata1 == metadata2
    assert len(first) == 10
    assert metadata1["inspected_nonblank_records"] == 100


def test_reservoir_sampling_skips_invalid_records(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        '{"sentence_id":"1","text":"valid"}\n'
        '{"sentence_id":"2","text":""}\n'
        'invalid-json\n'
        '{"sentence_id":"3","text":"also valid"}\n',
        encoding="utf-8",
    )

    records, metadata = audit._reservoir_sample(path, 10, 42)

    assert len(records) == 2
    assert metadata["invalid_records"] == 2
    assert metadata["inspected_nonblank_records"] == 4


def test_pearson_detects_perfect_correlation():
    result = audit._pearson(
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
    )
    assert result == pytest.approx(1.0)


def test_pearson_returns_none_for_constant_values():
    assert audit._pearson([1.0, 1.0], [2.0, 3.0]) is None


def test_feature_summary_identifies_constant_and_correlated_features():
    summary = audit._summarize_features(
        {
            "constant": [1.0, 1.0, 1.0],
            "near_constant": [1.0, 2.0, 1.0],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [2.0, 4.0, 6.0],
        },
        0.90,
    )

    assert "constant" in summary["constant_features"]
    assert "near_constant" in summary["near_constant_features"]
    assert len(summary["high_correlations"]) == 1
    assert summary["high_correlations"][0]["pearson_r"] == 1.0


def test_reference_overlap_uses_ids_and_normalized_text(tmp_path):
    reference = tmp_path / "reference.jsonl"
    write_jsonl(
        reference,
        [
            {"sentence_id": "10", "text": "The QUICK brown fox."},
            {"sentence_id": "11", "text": "Another sentence."},
        ],
    )

    ids, texts, metadata = audit._load_reference_keys(reference)

    overlap = audit._calculate_overlap(
        [
            {"sentence_id": 10, "text": "  THE   quick BROWN fox. "},
            {"sentence_id": "99", "text": "New sentence."},
        ],
        ids,
        texts,
    )

    assert metadata["unique_sentence_ids"] == 2
    assert overlap["sentence_id_overlap_count"] == 1
    assert overlap["normalized_text_overlap_count"] == 1


def test_audit_writes_report_and_reference_overlap(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    reference = tmp_path / "reference.jsonl"
    report = tmp_path / "reports" / "audit.json"

    records = [
        {"sentence_id": str(i), "text": f"sentence number {i}"}
        for i in range(8)
    ]
    write_jsonl(dataset, records)
    write_jsonl(reference, records[:4])

    monkeypatch.setattr(audit, "analyze_sentence_v2_2", fake_features)

    result = audit.run_v2_2_quality_audit(
        dataset,
        report,
        maximum_records=8,
        seed=42,
        reference_dataset_path=reference,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert result["records"] == 8
    assert payload["dataset"]["sha256"] == audit._sha256_file(dataset)
    assert payload["reference_dataset"]["sha256"] == audit._sha256_file(reference)
    assert payload["reference_dataset"]["overlap"]["sentence_id_overlap_count"] == 4
    assert payload["interpretation"]["production_approval"] is False


def test_audit_does_not_overwrite_existing_report(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    report = tmp_path / "audit.json"

    write_jsonl(dataset, [{"sentence_id": "1", "text": "example"}])
    report.write_text("preserve", encoding="utf-8")

    monkeypatch.setattr(audit, "analyze_sentence_v2_2", fake_features)

    with pytest.raises(FileExistsError):
        audit.run_v2_2_quality_audit(dataset, report)

    assert report.read_text(encoding="utf-8") == "preserve"


def test_audit_rejects_invalid_configuration(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    report = tmp_path / "audit.json"
    write_jsonl(dataset, [{"sentence_id": "1", "text": "example"}])

    with pytest.raises(ValueError):
        audit.run_v2_2_quality_audit(dataset, report, maximum_records=0)

    with pytest.raises(ValueError):
        audit.run_v2_2_quality_audit(
            dataset,
            report,
            correlation_threshold=0.0,
        )


def test_audit_rejects_empty_valid_dataset(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    report = tmp_path / "audit.json"
    dataset.write_text('{"sentence_id":"1","text":""}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        audit.run_v2_2_quality_audit(dataset, report)

    assert not report.exists()



def test_vectorized_correlations_match_reference_pearson():
    features = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "c": [5.0, 3.0, 4.0, 2.0, 1.0],
    }

    summary = audit._summarize_features(features, 0.0)

    expected = audit._pearson(features["a"], features["b"])
    actual = next(
        item["pearson_r"]
        for item in summary["high_correlations"]
        if {item["feature_a"], item["feature_b"]} == {"a", "b"}
    )

    assert actual == pytest.approx(expected, abs=1e-6)


def test_unequal_feature_lengths_use_reference_fallback():
    features = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],
        "c": [1.0, 2.0, 3.0],
    }

    summary = audit._summarize_features(features, 0.90)

    assert summary["high_correlations"] == [
        {
            "feature_a": "a",
            "feature_b": "b",
            "pearson_r": 1.0,
        }
    ]


def test_vectorized_correlation_skips_constant_features():
    summary = audit._summarize_features(
        {
            "constant": [5.0, 5.0, 5.0, 5.0],
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
        },
        0.90,
    )

    assert all(
        "constant" not in (
            item["feature_a"],
            item["feature_b"],
        )
        for item in summary["high_correlations"]
    )
    assert len(summary["high_correlations"]) == 1


def test_non_finite_feature_values_use_safe_fallback():
    summary = audit._summarize_features(
        {
            "a": [1.0, 2.0, float("nan")],
            "b": [2.0, 4.0, 6.0],
        },
        0.90,
    )

    assert summary["high_correlations"] == []


def test_correlation_threshold_boundary():
    features = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],
    }

    included = audit._summarize_features(features, 1.0)
    excluded = audit._summarize_features(features, 1.000001)

    assert len(included["high_correlations"]) == 1
    assert excluded["high_correlations"] == []
