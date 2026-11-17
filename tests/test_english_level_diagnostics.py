import json

import pytest

from vyrex_ai.data.english_level_diagnostics import (
    run_diagnostic_evaluation,
)


def _write_dataset(path, records):
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _record(sentence_id, text):
    return {
        "sentence_id": str(sentence_id),
        "text": text,
        "license": "CC0-1.0",
        "source_sha256": "test-source-hash",
        "source_archive": "test.tsv",
    }


def test_diagnostic_evaluation_writes_both_reports(tmp_path):
    dataset = tmp_path / "input.jsonl"
    report = tmp_path / "report.json"
    records = tmp_path / "records.jsonl"

    _write_dataset(dataset, [
        _record(1, "Students should complete their assignments."),
        _record(2, "The researcher who works here published a paper."),
    ])

    result = run_diagnostic_evaluation(
        dataset,
        report,
        records,
        maximum_records=10,
    )

    summary = json.loads(report.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in records.read_text(encoding="utf-8").splitlines()
    ]

    assert result["status"] == "completed"
    assert result["processed_records"] == 2
    assert summary["dataset"]["processed_records"] == 2
    assert len(rows) == 2
    assert summary["interpretation"]["production_approval"] is False
    assert summary["interpretation"]["human_annotations_available"] is False


def test_diagnostic_evaluation_respects_record_limit(tmp_path):
    dataset = tmp_path / "input.jsonl"
    report = tmp_path / "report.json"
    records = tmp_path / "records.jsonl"

    _write_dataset(dataset, [
        _record(1, "Students study English."),
        _record(2, "Researchers analyze evidence."),
        _record(3, "The system should be evaluated."),
    ])

    result = run_diagnostic_evaluation(
        dataset,
        report,
        records,
        maximum_records=2,
    )

    assert result["inspected_nonblank_records"] == 2
    assert result["processed_records"] == 2


def test_diagnostic_evaluation_rejects_zero_limit(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="maximum_records"):
        run_diagnostic_evaluation(
            dataset,
            tmp_path / "report.json",
            tmp_path / "records.jsonl",
            maximum_records=0,
        )


def test_diagnostic_evaluation_does_not_overwrite_report(tmp_path):
    dataset = tmp_path / "input.jsonl"
    report = tmp_path / "report.json"
    records = tmp_path / "records.jsonl"

    _write_dataset(dataset, [
        _record(1, "Students study English."),
    ])
    report.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_diagnostic_evaluation(dataset, report, records)

    assert report.read_text(encoding="utf-8") == "preserve"


def test_diagnostic_evaluation_rejects_same_output_path(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("", encoding="utf-8")

    output = tmp_path / "same.json"

    with pytest.raises(ValueError, match="must be different"):
        run_diagnostic_evaluation(dataset, output, output)


def test_diagnostic_evaluation_reports_malformed_records(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(
        '{"sentence_id":"1","text":"Students study English."}\n'
        '{invalid json}\n'
        '{"sentence_id":"2","text":""}\n',
        encoding="utf-8",
    )

    result = run_diagnostic_evaluation(
        dataset,
        tmp_path / "report.json",
        tmp_path / "records.jsonl",
    )

    assert result["processed_records"] == 1
    assert result["error_counts"]["malformed_json"] == 1
    assert result["error_counts"]["missing_or_invalid_text"] == 1


def test_diagnostic_evaluation_detects_duplicate_sentence_ids(tmp_path):
    dataset = tmp_path / "input.jsonl"

    _write_dataset(dataset, [
        _record(1, "Students study English."),
        _record(1, "Researchers analyze evidence."),
    ])

    result = run_diagnostic_evaluation(
        dataset,
        tmp_path / "report.json",
        tmp_path / "records.jsonl",
    )

    assert result["duplicate_sentence_ids"] == 1
    assert result["processed_records"] == 2


def test_diagnostic_evaluation_preserves_provisional_status(tmp_path):
    dataset = tmp_path / "input.jsonl"

    _write_dataset(dataset, [
        _record(1, "The system should have been evaluated."),
    ])

    run_diagnostic_evaluation(
        dataset,
        tmp_path / "report.json",
        tmp_path / "records.jsonl",
    )

    row = json.loads(
        (tmp_path / "records.jsonl").read_text(encoding="utf-8")
    )

    assert (
        row["v2"]["classification"]["classification_status"]
        == "provisional_not_human_approved"
    )
    assert row["v2"]["classification"]["confidence"] is None
    assert row["v2_1"]["auxiliary_chain_count"] == 1

def test_diagnostic_evaluation_reports_ratio_statistics(tmp_path):
    dataset = tmp_path / "input.jsonl"
    report = tmp_path / "report.json"
    records_path = tmp_path / "records.jsonl"

    _write_dataset(dataset, [
        _record(1, "Students study English."),
        _record(2, "The experienced researcher analyzes evidence carefully."),
    ])

    run_diagnostic_evaluation(
        dataset,
        report,
        records_path,
        maximum_records=10,
    )

    summary = json.loads(report.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]

    ratio_names = (
        "lexical_diversity",
        "long_word_ratio",
        "preposition_ratio",
        "nominalization_ratio",
    )

    statistics = summary["feature_comparison"]["v2_1_ratio_statistics"]

    for name in ratio_names:
        values = [float(row["v2_1"][name]) for row in rows]

        assert statistics[name]["mean"] == pytest.approx(
            sum(values) / len(values)
        )
        assert statistics[name]["minimum"] == pytest.approx(min(values))
        assert statistics[name]["maximum"] == pytest.approx(max(values))

    assert "v2_1_count_totals" in summary["feature_comparison"]
    assert "v2_1_feature_totals" not in summary["feature_comparison"]
