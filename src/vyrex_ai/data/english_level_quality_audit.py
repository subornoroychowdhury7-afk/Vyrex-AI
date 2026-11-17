"""VYREX V2.2 feature quality and performance audit."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import fmean, median, pvariance, pstdev

from vyrex_ai.data.english_level_classifier_v2_2 import (
    analyze_sentence_v2_2,
)

SCHEMA_VERSION = "1.0.0"
CORRELATION_THRESHOLD = 0.95
NEAR_CONSTANT_UNIQUE_RATIO = 0.01

def _flatten(result: dict) -> dict[str, float]:
    output = {}
    for group in ("lexical", "syntactic", "grammatical", "frequency"):
        for name, value in result[group].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    output[f"{group}.{name}"] = float(value)
    return output


def _distribution(values: list[float]) -> dict:
    counts = Counter(values)
    variance = pvariance(values) if len(values) > 1 else 0.0
    unique_count = len(counts)

    return {
        "count": len(values),
        "unique_count": unique_count,
        "unique_ratio": round(unique_count / len(values), 6) if values else 0.0,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": round(fmean(values), 8) if values else None,
        "median": round(median(values), 8) if values else None,
        "population_variance": round(variance, 10),
        "population_stddev": round(pstdev(values), 8) if values else 0.0,
        "most_common": [
            {"value": value, "count": count}
            for value, count in counts.most_common(10)
        ],
        "constant": unique_count <= 1,
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None

    left_mean = fmean(left)
    right_mean = fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]

    left_sum = sum(value * value for value in left_delta)
    right_sum = sum(value * value for value in right_delta)

    denominator = math.sqrt(left_sum * right_sum)
    if denominator == 0:
        return None

    return sum(
        a * b for a, b in zip(left_delta, right_delta)
    ) / denominator


def _correlations(columns: dict[str, list[float]]) -> list[dict]:
    names = sorted(columns)
    pairs = []

    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            correlation = _pearson(
                columns[left_name], columns[right_name]
            )
            if correlation is not None and abs(correlation) >= CORRELATION_THRESHOLD:
                pairs.append({
                    "feature_a": left_name,
                    "feature_b": right_name,
                    "pearson_r": round(correlation, 6),
                })

    return sorted(pairs, key=lambda item: abs(item["pearson_r"]), reverse=True)


def run_quality_audit(
    dataset_path: str | Path,
    report_path: str | Path,
    maximum_records: int = 1000,
) -> dict:
    source = Path(dataset_path)
    report = Path(report_path)

    if maximum_records < 1:
        raise ValueError("maximum_records must be positive")
    if not source.is_file():
        raise FileNotFoundError(source)
    if report.exists():
        raise FileExistsError(f"Report already exists: {report}")

    raw = source.read_bytes()
    records = []
    errors = []

    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if len(records) >= maximum_records:
            break
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError("Record must be an object with string text")
            records.append(record)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append({
                "line": line_number,
                "error": type(exc).__name__,
                "message": str(exc),
            })

    if not records:
        raise ValueError("Dataset contains no valid records")

    columns: dict[str, list[float]] = {}
    status_counts = Counter()
    first_duration = None
    warm_durations = []
    extraction_errors = []

    for index, record in enumerate(records):
        started = time.perf_counter()
        try:
            result = analyze_sentence_v2_2(record["text"])
            duration = time.perf_counter() - started

            if index == 0:
                first_duration = duration
            else:
                warm_durations.append(duration)

            status_counts[result["feature_status"]] += 1

            for name, value in _flatten(result).items():
                columns.setdefault(name, []).append(value)

        except Exception as exc:
            extraction_errors.append({
                "sentence_id": record.get("sentence_id"),
                "error": type(exc).__name__,
                "message": str(exc),
            })

    distributions = {
        name: _distribution(values)
        for name, values in sorted(columns.items())
    }

    sample_count = len(records)
    near_constant = [
        name for name, stats in distributions.items()
        if stats["constant"]
        or (
            stats["unique_ratio"] <= NEAR_CONSTANT_UNIQUE_RATIO
            and stats["population_variance"] < 1e-8
        )
    ]

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "dataset": str(source),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": sample_count,
        "successful_extractions": sum(status_counts.values()),
        "extraction_errors": extraction_errors,
        "input_errors": errors,
        "feature_distributions": distributions,
        "constant_features": [
            name for name, stats in distributions.items() if stats["constant"]
        ],
        "near_constant_features": near_constant,
        "high_correlation_pairs": _correlations(columns),
        "frequency_status_counts": dict(status_counts),
        "performance": {
            "first_record_seconds": first_duration,
            "warm_record_count": len(warm_durations),
            "warm_total_seconds": round(sum(warm_durations), 8),
            "warm_mean_seconds": (
                round(fmean(warm_durations), 8) if warm_durations else None
            ),
            "warm_median_seconds": (
                round(median(warm_durations), 8) if warm_durations else None
            ),
            "warm_records_per_second": (
                round(1 / fmean(warm_durations), 2)
                if warm_durations and fmean(warm_durations) > 0 else None
            ),
        },
        "configuration": {
            "correlation_threshold": CORRELATION_THRESHOLD,
            "near_constant_unique_ratio": NEAR_CONSTANT_UNIQUE_RATIO,
            "near_constant_variance_threshold": 1e-8,
            "frequency_resource": "optional wordfreq",
            "production_model_approved": False,
        },
    }

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return result
