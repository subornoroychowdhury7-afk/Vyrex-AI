"""Bounded, reproducible V2 versus V2.1 diagnostic evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vyrex_ai.data.english_level_classifier_v2 import (
    analyze_sentence,
)
from vyrex_ai.data.english_level_classifier_v2_1 import (
    analyze_sentence_v2_1,
)
from vyrex_ai.data.english_level_scoring_v2 import (
    classify_features,
)


DIAGNOSTIC_SCHEMA_VERSION = 1
V2_FEATURE_COUNT = 20
V2_1_ADDITIONAL_FEATURE_COUNT = 14


def _write_atomic(path: Path, writer) -> None:
    """Write an output atomically without overwriting existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")

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
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())

        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

        os.replace(temporary_path, path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def writer(handle):
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    _write_atomic(path, writer)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    def writer(handle):
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    _write_atomic(path, writer)


def run_diagnostic_evaluation(
    dataset_path: str | Path,
    report_path: str | Path,
    records_path: str | Path,
    maximum_records: int = 1000,
) -> dict[str, Any]:
    """Compare V2 and V2.1 diagnostics on a bounded dataset prefix."""

    if (
        isinstance(maximum_records, bool)
        or not isinstance(maximum_records, int)
        or maximum_records < 1
    ):
        raise ValueError("maximum_records must be a positive integer")

    source = Path(dataset_path)
    report = Path(report_path)
    records_output = Path(records_path)

    if not source.is_file():
        raise FileNotFoundError(source)

    resolved_paths = {
        source.resolve(),
        report.resolve(),
        records_output.resolve(),
    }

    if len(resolved_paths) != 3:
        raise ValueError(
            "Dataset, report, and record output paths must be different"
        )

    if report.exists():
        raise FileExistsError(f"Report already exists: {report}")

    if records_output.exists():
        raise FileExistsError(
            f"Record output already exists: {records_output}"
        )

    prefix_digest = hashlib.sha256()
    records: list[dict] = []
    errors = Counter()
    seen_ids: set[str] = set()
    class_counts = Counter()
    feature_totals = Counter()
    ratio_values = {
        "lexical_diversity": [],
        "long_word_ratio": [],
        "preposition_ratio": [],
        "nominalization_ratio": [],
    }
    processed_lines = 0
    inspected_nonblank = 0
    duplicate_ids = 0

    with source.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue

            if inspected_nonblank >= maximum_records:
                break

            inspected_nonblank += 1
            prefix_digest.update(raw_line)

            try:
                item = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                errors["malformed_json"] += 1
                continue

            if not isinstance(item, dict):
                errors["non_object_record"] += 1
                continue

            text = item.get("text")

            if not isinstance(text, str) or not text.strip():
                errors["missing_or_invalid_text"] += 1
                continue

            sentence_id = item.get("sentence_id")

            if sentence_id is not None:
                normalized_id = str(sentence_id)
                if normalized_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(normalized_id)

            try:
                v2_features = analyze_sentence(text)
                v2_classification = classify_features(v2_features)
                v2_1_features = analyze_sentence_v2_1(text)
            except (TypeError, ValueError) as exc:
                errors["analysis_error"] += 1
                continue

            processed_lines += 1
            class_counts[v2_classification["predicted_level"]] += 1

            for name, value in v2_1_features.items():
                if name == "baseline_feature_count":
                    continue

                if name in ratio_values:
                    ratio_values[name].append(float(value))
                else:
                    feature_totals[name] += value

            records.append({
                "sentence_id": sentence_id,
                "text": text,
                "provenance": {
                    "license": item.get("license"),
                    "source_sha256": item.get("source_sha256"),
                    "source_archive": item.get("source_archive"),
                    "author_metadata_status": item.get(
                        "author_metadata_status"
                    ),
                },
                "v2": {
                    "features": v2_features,
                    "classification": v2_classification,
                },
                "v2_1": v2_1_features,
                "comparison": {
                    "shared_word_count": {
                        "v2": v2_features["word_count"],
                        "v2_1": v2_1_features["word_count"],
                        "consistent": (
                            v2_features["word_count"]
                            == v2_1_features["word_count"]
                        ),
                    },
                    "additional_diagnostic_feature_count": (
                        len(v2_1_features) - 1
                    ),
                },
            })

    if not records:
        raise ValueError(
            "No valid text records were available for evaluation"
        )

    word_count_consistent = all(
        record["comparison"]["shared_word_count"]["consistent"]
        for record in records
    )

    summary = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(source),
            "sampled_prefix_sha256": prefix_digest.hexdigest(),
            "hash_scope": "inspected_nonblank_record_prefix",
            "maximum_records": maximum_records,
            "inspected_nonblank_records": inspected_nonblank,
            "processed_records": processed_lines,
            "error_counts": dict(errors),
            "duplicate_sentence_ids": duplicate_ids,
            "class_counts": dict(class_counts),
        },
        "feature_comparison": {
            "v2_baseline_feature_count": V2_FEATURE_COUNT,
            "v2_1_additional_diagnostic_feature_count": (
                V2_1_ADDITIONAL_FEATURE_COUNT
            ),
            "v2_1_returned_key_count": (
                V2_1_ADDITIONAL_FEATURE_COUNT + 1
            ),
            "shared_word_count_consistent": word_count_consistent,
            "v2_1_count_totals": dict(feature_totals),
            "v2_1_ratio_statistics": {
                name: {
                    "mean": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in ratio_values.items()
                if values
            },
            "comparison_policy": (
                "Only identically defined measurements are compared. "
                "Similarly named but differently defined features are "
                "reported independently."
            ),
        },
        "environment": {
            "python_version": platform.python_version(),
        },
        "interpretation": {
            "human_annotations_available": False,
            "false_positive_rate_calculated": False,
            "false_negative_rate_calculated": False,
            "confidence_calibrated": False,
            "production_approval": False,
            "note": (
                "This is a bounded diagnostic comparison, not a "
                "human-labelled accuracy evaluation. V2 classifications "
                "remain provisional. V2.1 features are experimental."
            ),
        },
    }

    _write_json(report, summary)

    try:
        _write_jsonl(records_output, records)
    except Exception:
        # Preserve no-overwrite semantics while avoiding an orphaned report.
        report.unlink(missing_ok=True)
        raise

    return {
        "status": "completed",
        "processed_records": processed_lines,
        "inspected_nonblank_records": inspected_nonblank,
        "error_counts": dict(errors),
        "duplicate_sentence_ids": duplicate_ids,
        "report_path": str(report),
        "records_path": str(records_output),
        "sampled_prefix_sha256": prefix_digest.hexdigest(),
        "production_approval": False,
    }

