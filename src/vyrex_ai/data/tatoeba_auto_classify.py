"""Streaming batch classification for Tatoeba English sentences."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from vyrex_ai.data.english_level_classifier import classify_sentence


def classify_tatoeba_batch(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    batch_size: int = 1000,
) -> dict:
    """Classify records incrementally without modifying the source dataset."""

    source = Path(input_path)
    destination = Path(output_directory)

    if not source.is_file():
        raise FileNotFoundError(source)

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")

    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    ))

    level_counts = Counter()
    review_counts = Counter()
    total = 0

    try:
        predictions_path = staging / "predictions.jsonl"
        review_path = staging / "review_queue.jsonl"

        with (
            source.open("r", encoding="utf-8") as input_handle,
            predictions_path.open("w", encoding="utf-8", newline="\n") as predictions,
            review_path.open("w", encoding="utf-8", newline="\n") as review,
        ):
            for line_number, line in enumerate(input_handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON at line {line_number}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record at line {line_number} is not an object"
                    )

                if not isinstance(record.get("text"), str):
                    raise ValueError(
                        f"Missing sentence text at line {line_number}"
                    )

                result = classify_sentence(record["text"])

                output = dict(record)
                output["classification"] = result

                serialized = json.dumps(output, ensure_ascii=False) + "\n"
                predictions.write(serialized)

                level = result["predicted_level"]
                level_counts[level] += 1
                total += 1

                if result["review_required"]:
                    review.write(serialized)
                    review_counts[level] += 1

                if total % batch_size == 0:
                    predictions.flush()
                    review.flush()

            predictions.flush()
            review.flush()
            os.fsync(predictions.fileno())
            os.fsync(review.fileno())

        summary = {
            "schema_version": 1,
            "source_file": str(source.resolve()),
            "total_processed": total,
            "batch_size": batch_size,
            "predictions_file": "predictions.jsonl",
            "review_queue_file": "review_queue.jsonl",
            "predictions_by_level": dict(sorted(level_counts.items())),
            "review_required_total": sum(review_counts.values()),
            "review_required_by_level": dict(sorted(review_counts.items())),
            "automatic_human_approval": False,
            "confidence_type": "uncalibrated_heuristic",
            "training_eligible": False,
            "original_source_modified": False,
        }

        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        os.replace(staging, destination)
        return summary

    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
