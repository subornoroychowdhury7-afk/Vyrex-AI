from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import tempfile
import time

import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vyrex_ai.data.english_level_classifier_v2_2 import (
    FEATURE_SCHEMA_VERSION,
    analyze_sentence_v2_2,
)


AUDIT_SCHEMA_VERSION = 1
CORRELATION_THRESHOLD = 0.90
NEAR_CONSTANT_UNIQUE_LIMIT = 2


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Audit report already exists: {path}")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if path.exists():
            raise FileExistsError(f"Audit report already exists: {path}")

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_features(result: dict[str, Any]) -> dict[str, float]:
    flattened = {}
    for group in ("lexical", "syntactic", "grammatical", "frequency"):
        for name, value in result[group].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    flattened[f"{group}.{name}"] = float(value)
    return flattened


def _reservoir_sample(
    dataset_path: Path,
    maximum_records: int,
    seed: int,
) -> tuple[list[dict], dict[str, Any]]:
    rng = random.Random(seed)
    reservoir = []
    digest = hashlib.sha256()
    inspected = 0
    valid = 0
    invalid = 0

    with dataset_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue

            digest.update(line.encode("utf-8"))
            inspected += 1

            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Record must be a JSON object")
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Missing nonblank text")
            except (json.JSONDecodeError, ValueError):
                invalid += 1
                continue

            valid += 1
            if len(reservoir) < maximum_records:
                reservoir.append(record)
            else:
                index = rng.randrange(valid)
                if index < maximum_records:
                    reservoir[index] = record

    return reservoir, {
        "inspected_nonblank_records": inspected,
        "invalid_records": invalid,
        "sample_records": len(reservoir),
        "sample_seed": seed,
        "inspected_stream_sha256": digest.hexdigest(),
    }



def _normalize_sentence(text: str) -> str:
    import unicodedata

    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().split()
    )


def _load_reference_keys(
    reference_path: Path,
) -> tuple[set[str], set[str], dict[str, Any]]:
    sentence_ids = set()
    normalized_texts = set()
    inspected = 0
    invalid = 0

    with reference_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue

            inspected += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected JSON object")

                sentence_id = record.get("sentence_id")
                text = record.get("text")

                if sentence_id is not None:
                    sentence_ids.add(str(sentence_id))

                if isinstance(text, str) and text.strip():
                    normalized_texts.add(_normalize_sentence(text))
            except (json.JSONDecodeError, ValueError):
                invalid += 1

    return sentence_ids, normalized_texts, {
        "inspected_nonblank_records": inspected,
        "invalid_records": invalid,
        "unique_sentence_ids": len(sentence_ids),
        "unique_normalized_texts": len(normalized_texts),
        "sha256": _sha256_file(reference_path),
    }


def _calculate_overlap(
    records: list[dict],
    reference_ids: set[str],
    reference_texts: set[str],
) -> dict[str, Any]:
    sample_ids = {
        str(record["sentence_id"])
        for record in records
        if record.get("sentence_id") is not None
    }
    sample_texts = {
        _normalize_sentence(record["text"])
        for record in records
        if isinstance(record.get("text"), str)
        and record["text"].strip()
    }

    id_overlap = sample_ids & reference_ids
    text_overlap = sample_texts & reference_texts

    return {
        "sample_unique_sentence_ids": len(sample_ids),
        "sample_unique_normalized_texts": len(sample_texts),
        "sentence_id_overlap_count": len(id_overlap),
        "normalized_text_overlap_count": len(text_overlap),
        "sentence_id_overlap_ratio": (
            len(id_overlap) / len(sample_ids) if sample_ids else 0.0
        ),
        "normalized_text_overlap_ratio": (
            len(text_overlap) / len(sample_texts) if sample_texts else 0.0
        ),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None

    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)

    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]

    sum_left = sum(value * value for value in centered_left)
    sum_right = sum(value * value for value in centered_right)

    denominator = math.sqrt(sum_left * sum_right)
    if denominator == 0:
        return None

    return sum(
        a * b for a, b in zip(centered_left, centered_right)
    ) / denominator


def _summarize_features(
    feature_values: dict[str, list[float]],
    correlation_threshold: float,
) -> dict[str, Any]:
    distributions = {}
    constant = []
    near_constant = []

    for name, values in sorted(feature_values.items()):
        unique_count = len(set(values))
        variance = statistics.pvariance(values) if values else 0.0

        distributions[name] = {
            "count": len(values),
            "unique_count": unique_count,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "variance": variance,
            "standard_deviation": math.sqrt(variance),
        }

        if unique_count <= 1:
            constant.append(name)
        elif unique_count <= NEAR_CONSTANT_UNIQUE_LIMIT:
            near_constant.append(name)

    names = sorted(feature_values)
    correlations = []

    lengths = {len(feature_values[name]) for name in names}
    can_vectorize = (
        len(lengths) == 1
        and next(iter(lengths), 0) >= 2
        and all(
            np.isfinite(feature_values[name]).all()
            for name in names
        )
    )

    if can_vectorize and len(names) >= 2:
        matrix = np.asarray(
            [feature_values[name] for name in names],
            dtype=np.float64,
        )

        valid_indices = np.flatnonzero(matrix.std(axis=1) > 0)

        if len(valid_indices) >= 2:
            correlation_matrix = np.corrcoef(matrix[valid_indices])
            upper_rows, upper_cols = np.triu_indices(
                len(valid_indices),
                k=1,
            )

            selected = (
                np.abs(correlation_matrix[upper_rows, upper_cols])
                >= correlation_threshold
            )

            for row, col in zip(
                upper_rows[selected],
                upper_cols[selected],
            ):
                correlations.append({
                    "feature_a": names[valid_indices[row]],
                    "feature_b": names[valid_indices[col]],
                    "pearson_r": round(
                        float(correlation_matrix[row, col]),
                        6,
                    ),
                })
    else:
        for index, name_a in enumerate(names):
            for name_b in names[index + 1:]:
                correlation = _pearson(
                    feature_values[name_a],
                    feature_values[name_b],
                )

                if (
                    correlation is not None
                    and abs(correlation) >= correlation_threshold
                ):
                    correlations.append({
                        "feature_a": name_a,
                        "feature_b": name_b,
                        "pearson_r": round(correlation, 6),
                    })

    correlations.sort(
        key=lambda item: abs(item["pearson_r"]),
        reverse=True,
    )

    return {
        "feature_count": len(distributions),
        "distributions": distributions,
        "constant_features": constant,
        "near_constant_features": near_constant,
        "high_correlations": correlations,
        "correlation_threshold": correlation_threshold,
    }


def _benchmark(
    texts: list[str],
) -> dict[str, Any]:
    if not texts:
        raise ValueError("No valid text records available for benchmarking")

    start = time.perf_counter()
    analyze_sentence_v2_2(texts[0])
    first_seconds = time.perf_counter() - start

    warm_times = []
    for text in texts[1:]:
        start = time.perf_counter()
        analyze_sentence_v2_2(text)
        warm_times.append(time.perf_counter() - start)

    total = sum(warm_times)

    return {
        "first_call_seconds": first_seconds,
        "warm_record_count": len(warm_times),
        "warm_total_seconds": total,
        "warm_mean_seconds": (
            statistics.fmean(warm_times) if warm_times else 0.0
        ),
        "warm_median_seconds": (
            statistics.median(warm_times) if warm_times else 0.0
        ),
        "warm_records_per_second": (
            len(warm_times) / total if total > 0 else None
        ),
    }


def run_v2_2_quality_audit(
    dataset_path: str | Path,
    report_path: str | Path,
    maximum_records: int = 5000,
    seed: int = 42,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    reference_dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(dataset_path)
    destination = Path(report_path)

    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        raise ValueError("Report path cannot overwrite source dataset")

    reference_path = (
        Path(reference_dataset_path)
        if reference_dataset_path is not None
        else None
    )
    if reference_path is not None:
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        if reference_path.resolve() in {
            source.resolve(),
            destination.resolve(),
        }:
            raise ValueError(
                "Reference dataset must differ from source and report paths"
            )

    if destination.exists():
        raise FileExistsError(f"Audit report already exists: {destination}")
    if maximum_records < 1:
        raise ValueError("maximum_records must be positive")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")

    dataset_hash = _sha256_file(source)
    records, sampling = _reservoir_sample(
        source,
        maximum_records,
        seed,
    )

    overlap = None
    reference_metadata = None
    if reference_path is not None:
        reference_ids, reference_texts, reference_metadata = (
            _load_reference_keys(reference_path)
        )
        overlap = _calculate_overlap(
            records,
            reference_ids,
            reference_texts,
        )

    texts = [record["text"] for record in records]
    if not texts:
        raise ValueError("No valid text records available")

    # Benchmark before feature extraction so this process's first
    # analysis call is not preceded by the audit's feature pass.
    performance = _benchmark(texts)

    feature_values: dict[str, list[float]] = {}
    errors = Counter()

    for text in texts:
        try:
            result = analyze_sentence_v2_2(text)
            for name, value in _flatten_features(result).items():
                feature_values.setdefault(name, []).append(value)
        except (TypeError, ValueError, KeyError):
            errors["feature_extraction_error"] += 1

    if not feature_values:
        raise ValueError("No records could be analyzed")

    summary = _summarize_features(
        feature_values,
        correlation_threshold,
    )

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(source),
            "sha256": dataset_hash,
            **sampling,
            "processed_records": len(texts),
            "hash_scope": "complete_source_file",
        },
        "reference_dataset": (
            {
                "path": str(reference_path),
                **reference_metadata,
                "overlap": overlap,
            }
            if reference_path is not None
            else None
        ),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_quality": summary,
        "performance": performance,
        "errors": dict(errors),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "interpretation": {
            "confidence_calibrated": False,
            "human_labels_used": False,
            "production_approval": False,
            "note": (
                "Feature quality diagnostics only; no proficiency "
                "classification performance or causal conclusions."
            ),
        },
    }

    _write_json_atomic(destination, report)

    return {
        "status": "completed",
        "records": len(texts),
        "errors": sum(errors.values()),
        "constant": summary["constant_features"],
        "near_constant": summary["near_constant_features"],
        "correlations": summary["high_correlations"],
        "performance": performance,
        "report_path": str(destination),
        "dataset_sha256": dataset_hash,
        "reference_overlap": overlap,
    }



