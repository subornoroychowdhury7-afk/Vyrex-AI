import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from vyrex_ai.data.english_level_ml_experiment import (
    run_ml_experiment,
    _sha256_file,
)


@dataclass
class Example:
    text: str
    label: str
    features: tuple[float, ...]


def make_dataset(path: Path, counts=None):
    counts = counts or {
        "basic_english": 10,
        "elementary_intermediate_english": 10,
        "upper_intermediate_english": 10,
        "advanced_english": 10,
    }

    records = []

    for label, count in counts.items():
        for index in range(count):
            records.append({
                "sentence_id": f"{label}-{index}",
                "text": f"{label} distinct sentence number {index}",
                "review_status": "approved",
                "assigned_level": label,
                "rationale": "Reviewed sentence structure and vocabulary.",
                "reviewer": "Synthetic Test Reviewer",
                "reviewed_at": "2026-11-13T10:00:00+00:00",
            })

    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_dataset_hash_is_sha256(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text("synthetic data", encoding="utf-8")

    digest = _sha256_file(path)

    assert len(digest) == 64
    assert digest == _sha256_file(path)


def test_experiment_rejects_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_ml_experiment(
            tmp_path / "missing.jsonl",
            tmp_path / "report.json",
        )


def test_experiment_rejects_invalid_fold_count(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    make_dataset(dataset)

    with pytest.raises(ValueError, match="n_splits"):
        run_ml_experiment(
            dataset,
            tmp_path / "report.json",
            n_splits=1,
        )


def test_experiment_rejects_missing_class(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    make_dataset(dataset, {
        "basic_english": 10,
        "upper_intermediate_english": 10,
        "advanced_english": 10,
    })

    with pytest.raises(ValueError, match="Missing"):
        run_ml_experiment(
            dataset,
            tmp_path / "report.json",
            n_splits=5,
        )


def test_experiment_rejects_insufficient_independent_groups(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    make_dataset(dataset, {
        "basic_english": 4,
        "elementary_intermediate_english": 10,
        "upper_intermediate_english": 10,
        "advanced_english": 10,
    })

    with pytest.raises(ValueError, match="Insufficient independent"):
        run_ml_experiment(
            dataset,
            tmp_path / "report.json",
            n_splits=5,
        )


def test_experiment_will_not_overwrite_dataset(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    make_dataset(dataset)

    original = dataset.read_bytes()

    with pytest.raises(ValueError, match="overwrite"):
        run_ml_experiment(dataset, dataset)

    assert dataset.read_bytes() == original


def test_experiment_will_not_overwrite_report(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    report = tmp_path / "report.json"

    make_dataset(dataset)
    report.write_text("existing report", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_ml_experiment(dataset, report)

    assert report.read_text(encoding="utf-8") == "existing report"


def test_report_writer_produces_valid_json(tmp_path):
    from vyrex_ai.data.english_level_ml_experiment import (
        _write_json_atomic,
    )

    report = tmp_path / "report.json"
    payload = {"schema_version": 1, "status": "test"}

    _write_json_atomic(report, payload)

    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_synthetic_experiment_completes(tmp_path):
    dataset = tmp_path / "synthetic.jsonl"
    report = tmp_path / "experiment.json"

    make_dataset(dataset)

    result = run_ml_experiment(
        dataset,
        report,
        n_splits=5,
        random_state=42,
    )

    assert result["status"] == "completed"
    assert result["sample_count"] == 40
    assert result["fold_count"] == 5
    assert set(result["model_names"]) == {
        "logistic_regression",
        "linear_svm",
        "random_forest",
    }

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert payload["evaluation"]["human_approved_labels_only"] is True
    assert payload["evaluation"]["confidence_calibration"] == "not_performed"
    assert payload["interpretation"]["production_model_approved"] is False
    assert len(payload["dataset"]["sha256"]) == 64
    assert len(payload["results"]["out_of_fold_predictions"]) == 3
