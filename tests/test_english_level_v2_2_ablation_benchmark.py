import json

import pytest

from vyrex_ai.data.english_level_v2_2_ablation_benchmark import (
    run_v2_2_ablation_benchmark,
)


@pytest.fixture
def sample_dataset(tmp_path):
    path = tmp_path / "sentences.jsonl"
    records = [
        {"sentence_id": str(i), "text": text}
        for i, text in enumerate([
            "The researchers carefully evaluated the results.",
            "Children enjoy playing outside.",
            "Although it was raining, we continued working.",
            "The experiment was carefully designed.",
            "She could have completed the assignment earlier.",
        ])
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_benchmark_completes_with_expected_configurations(
    sample_dataset, tmp_path
):
    report_path = tmp_path / "benchmark.json"

    result = run_v2_2_ablation_benchmark(
        sample_dataset,
        report_path,
        maximum_records=5,
        seed=42,
    )

    assert result["status"] == "completed"
    assert result["records"] == 5
    assert result["errors"] == 0
    assert result["configurations"] == {
        "full": 21,
        "constant_removed": 20,
        "redundancy_reduced": 17,
    }
    assert report_path.is_file()


def test_report_contains_separate_performance_measurements(
    sample_dataset, tmp_path
):
    report_path = tmp_path / "benchmark.json"

    run_v2_2_ablation_benchmark(
        sample_dataset,
        report_path,
        maximum_records=5,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "extraction" in report["performance"]
    assert "filtering" in report["performance"]
    assert set(report["performance"]["filtering"]) == {
        "full",
        "constant_removed",
        "redundancy_reduced",
    }


def test_report_preserves_experimental_boundaries(
    sample_dataset, tmp_path
):
    report_path = tmp_path / "benchmark.json"

    run_v2_2_ablation_benchmark(
        sample_dataset,
        report_path,
        maximum_records=5,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["interpretation"]["human_labels_used"] is False
    assert report["interpretation"]["confidence_calibrated"] is False
    assert report["interpretation"]["predictive_performance_evaluated"] is False
    assert report["interpretation"]["production_approval"] is False


def test_existing_report_is_never_overwritten(
    sample_dataset, tmp_path
):
    report_path = tmp_path / "benchmark.json"
    report_path.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_v2_2_ablation_benchmark(
            sample_dataset,
            report_path,
            maximum_records=5,
        )

    assert report_path.read_text(encoding="utf-8") == "preserve"


def test_invalid_correlation_threshold_is_rejected(
    sample_dataset, tmp_path
):
    with pytest.raises(ValueError):
        run_v2_2_ablation_benchmark(
            sample_dataset,
            tmp_path / "benchmark.json",
            correlation_threshold=0,
        )
