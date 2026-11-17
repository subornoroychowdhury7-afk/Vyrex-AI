"""Reproducible V2.2 feature ablation benchmark."""

from __future__ import annotations

import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vyrex_ai.data.english_level_classifier_v2_2 import (
    FEATURE_SCHEMA_VERSION,
    analyze_sentence_v2_2,
)
from vyrex_ai.data.english_level_v2_2_ablation import (
    FEATURE_COUNTS,
    REMOVED_FEATURES,
    apply_v2_2_ablation,
)
from vyrex_ai.data.english_level_v2_2_audit import (
    _flatten_features,
    _reservoir_sample,
    _sha256_file,
    _summarize_features,
    _write_json_atomic,
)


BENCHMARK_SCHEMA_VERSION = 1
CONFIGURATIONS = tuple(REMOVED_FEATURES)


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    total = sum(samples)
    return {
        "record_count": len(samples),
        "total_seconds": total,
        "mean_seconds": statistics.fmean(samples) if samples else 0.0,
        "median_seconds": statistics.median(samples) if samples else 0.0,
        "records_per_second": (
            len(samples) / total if total > 0 else None
        ),
    }


def _extract_features(
    texts: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], Counter]:
    results = []
    timings = []
    errors = Counter()

    for text in texts:
        start = time.perf_counter()
        try:
            result = analyze_sentence_v2_2(text)
            results.append(result)
        except (TypeError, ValueError, KeyError):
            errors["feature_extraction_error"] += 1
        finally:
            timings.append(time.perf_counter() - start)

    return results, {
        "first_call_seconds": timings[0] if timings else 0.0,
        "warm": _timing_summary(timings[1:]),
        "total": _timing_summary(timings),
    }, errors


def _benchmark_filtering(
    feature_results: list[dict[str, Any]],
) -> dict[str, Any]:
    timings = {name: [] for name in CONFIGURATIONS}

    for result in feature_results:
        for name in CONFIGURATIONS:
            start = time.perf_counter()
            apply_v2_2_ablation(result, name)
            timings[name].append(time.perf_counter() - start)

    return {
        name: _timing_summary(values)
        for name, values in timings.items()
    }


def _configuration_summaries(
    feature_results: list[dict[str, Any]],
    correlation_threshold: float,
) -> dict[str, Any]:
    feature_values = {
        name: {}
        for name in CONFIGURATIONS
    }

    removed_features = REMOVED_FEATURES

    for result in feature_results:
        flattened = _flatten_features(result)

        for configuration in CONFIGURATIONS:
            excluded = removed_features[configuration]
            values = feature_values[configuration]

            for feature_name, value in flattened.items():
                if feature_name not in excluded:
                    values.setdefault(feature_name, []).append(value)

    configurations = {}

    for name in CONFIGURATIONS:
        summary = _summarize_features(
            feature_values[name],
            correlation_threshold,
        )

        expected = FEATURE_COUNTS[name]

        if summary["feature_count"] != expected:
            raise ValueError(
                f"{name}: expected {expected} features, "
                f"found {summary['feature_count']}"
            )

        configurations[name] = {
            "feature_count": summary["feature_count"],
            "removed_features": sorted(REMOVED_FEATURES[name]),
            "feature_quality": summary,
        }

    return configurations
def run_v2_2_ablation_benchmark(
    dataset_path: str | Path,
    report_path: str | Path,
    maximum_records: int = 5000,
    seed: int = 42,
    correlation_threshold: float = 0.90,
) -> dict[str, Any]:
    source = Path(dataset_path)
    destination = Path(report_path)

    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        raise ValueError("Report cannot overwrite source dataset")
    if destination.exists():
        raise FileExistsError(f"Report already exists: {destination}")
    if maximum_records < 1:
        raise ValueError("maximum_records must be positive")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")

    records, sampling = _reservoir_sample(
        source,
        maximum_records,
        seed,
    )
    texts = [record["text"] for record in records]

    if not texts:
        raise ValueError("No valid text records available")

    feature_results, extraction_performance, errors = _extract_features(texts)

    if not feature_results:
        raise ValueError("No records could be analyzed")

    filtering_performance = _benchmark_filtering(feature_results)

    summarization_start = time.perf_counter()
    configurations = _configuration_summaries(
        feature_results,
        correlation_threshold,
    )
    summarization_seconds = time.perf_counter() - summarization_start

    summarization_performance = {
        "record_count": len(feature_results),
        "total_seconds": summarization_seconds,
        "mean_seconds_per_record": (
            summarization_seconds / len(feature_results)
        ),
        "records_per_second": (
            len(feature_results) / summarization_seconds
            if summarization_seconds > 0 else 0.0
        ),
    }

    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(source),
            "sha256": _sha256_file(source),
            **sampling,
            "processed_records": len(feature_results),
            "hash_scope": "complete_source_file",
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "configurations": configurations,
        "performance": {
            "extraction": extraction_performance,
            "filtering": filtering_performance,
            "summarization": summarization_performance,
            "note": (
                "Filtering is measured separately. All configurations "
                "share one extraction pass; filtering measurements do "
                "not represent end-to-end inference speed."
            ),
        },
        "errors": dict(errors),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "interpretation": {
            "human_labels_used": False,
            "confidence_calibrated": False,
            "predictive_performance_evaluated": False,
            "production_approval": False,
        },
    }

    _write_json_atomic(destination, report)

    return {
        "status": "completed",
        "records": len(feature_results),
        "errors": sum(errors.values()),
        "configurations": {
            name: value["feature_count"]
            for name, value in configurations.items()
        },
        "extraction": extraction_performance,
        "filtering": filtering_performance,
        "summarization": summarization_performance,
        "report_path": str(destination),
        "dataset_sha256": report["dataset"]["sha256"],
    }



