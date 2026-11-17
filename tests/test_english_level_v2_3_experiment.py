import json
from types import SimpleNamespace

import pytest

from vyrex_ai.data import english_level_v2_3_experiment as experiment
from vyrex_ai.data.english_level_classifier_v2 import LEVELS


def make_examples(per_class=2):
    return [
        SimpleNamespace(
            sentence_id=f"{level}-{index}",
            text=f"Unique English sentence for {level} number {index}.",
            label=level,
            rationale="This label is supported by linguistic evidence.",
            reviewer="test-reviewer",
            reviewed_at="2026-11-13T10:00:00+05:30",
        )
        for level in LEVELS
        for index in range(per_class)
    ]


def install_loader(monkeypatch, examples):
    monkeypatch.setattr(
        experiment,
        "load_human_labelled_data",
        lambda *args, **kwargs: {
            "total_records": len(examples),
            "eligible_records": len(examples),
            "excluded_records": 0,
            "examples": examples,
        },
    )


def test_missing_class_prevents_evaluation(tmp_path, monkeypatch):
    dataset = tmp_path / "labels.jsonl"
    report = tmp_path / "report.json"
    dataset.write_text("{}\n", encoding="utf-8")

    examples = [
        item for item in make_examples()
        if item.label != LEVELS[1]
    ]
    install_loader(monkeypatch, examples)

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("Evaluation must not run with a missing class")

    monkeypatch.setattr(
        experiment,
        "run_v2_3_feature_validation",
        unexpected_evaluation,
    )

    with pytest.raises(ValueError, match="Missing"):
        experiment.run_v2_3_experiment(dataset, report, n_splits=2)

    assert not report.exists()


def test_insufficient_groups_prevent_evaluation(tmp_path, monkeypatch):
    dataset = tmp_path / "labels.jsonl"
    report = tmp_path / "report.json"
    dataset.write_text("{}\n", encoding="utf-8")

    examples = make_examples(per_class=1)
    install_loader(monkeypatch, examples)

    with pytest.raises(ValueError, match="Insufficient independent"):
        experiment.run_v2_3_experiment(dataset, report, n_splits=2)

    assert not report.exists()


def test_success_writes_v23_report_atomically(tmp_path, monkeypatch):
    dataset = tmp_path / "labels.jsonl"
    report = tmp_path / "report.json"
    dataset.write_text("{}\n", encoding="utf-8")

    examples = make_examples()
    install_loader(monkeypatch, examples)

    monkeypatch.setattr(
        experiment,
        "run_v2_3_feature_validation",
        lambda *args, **kwargs: {
            "schema_version": "2.3.0",
            "status": "completed",
            "sample_count": len(examples),
        },
    )

    result = experiment.run_v2_3_experiment(
        dataset,
        report,
        n_splits=2,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert payload["schema_version"] == "2.3.0"
    assert payload["evaluation"]["human_approved_labels_only"] is True
    assert payload["interpretation"]["production_model_approved"] is False
    assert payload["feature_configurations"]["full"]["feature_count"] == 21
    assert payload["feature_configurations"]["constant_removed"]["feature_count"] == 20
    assert payload["feature_configurations"]["redundancy_reduced"]["feature_count"] == 17


def test_existing_report_is_never_overwritten(tmp_path, monkeypatch):
    dataset = tmp_path / "labels.jsonl"
    report = tmp_path / "report.json"
    dataset.write_text("{}\n", encoding="utf-8")
    report.write_text("preserve", encoding="utf-8")

    install_loader(monkeypatch, make_examples())

    with pytest.raises(FileExistsError):
        experiment.run_v2_3_experiment(dataset, report, n_splits=2)

    assert report.read_text(encoding="utf-8") == "preserve"
