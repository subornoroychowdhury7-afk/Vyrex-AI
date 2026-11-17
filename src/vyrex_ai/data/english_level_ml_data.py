"""Human-labelled dataset preparation for VYREX English-level ML."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from vyrex_ai.data.english_level_classifier_v2 import LEVELS
from vyrex_ai.data.english_level_ml import extract_feature_vector


@dataclass(frozen=True)
class HumanLabelledExample:
    sentence_id: str
    text: str
    label: str
    rationale: str
    reviewer: str
    reviewed_at: str
    features: tuple[float, ...]


def _valid_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None
    except (ValueError, TypeError):
        return False


def load_human_labelled_data(
    path: str | Path,
    minimum_rationale_words: int = 3,
) -> dict:
    """Load approved human labels and separately report excluded records."""

    if (
        isinstance(minimum_rationale_words, bool)
        or not isinstance(minimum_rationale_words, int)
        or minimum_rationale_words < 1
    ):
        raise ValueError("minimum_rationale_words must be positive")

    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(source)

    examples = []
    excluded = []
    seen_ids = set()
    total = 0

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record on line {line_number} must be an object"
                )

            sentence_id = record.get("sentence_id")

            if not isinstance(sentence_id, str) or not sentence_id.strip():
                raise ValueError(
                    f"Missing sentence_id on line {line_number}"
                )

            if sentence_id in seen_ids:
                raise ValueError(
                    f"Duplicate sentence_id: {sentence_id}"
                )

            seen_ids.add(sentence_id)

            def exclude(reason):
                excluded.append({
                    "sentence_id": sentence_id,
                    "reason": reason,
                })

            if record.get("review_status") != "approved":
                exclude("not human-approved")
                continue

            text = record.get("text")
            label = record.get("assigned_level")
            rationale = record.get("rationale")
            reviewer = record.get("reviewer")
            reviewed_at = record.get("reviewed_at")

            if not isinstance(text, str) or not text.strip():
                exclude("missing sentence text")
                continue

            if label not in LEVELS:
                exclude("missing or unsupported human label")
                continue

            if not isinstance(rationale, str):
                exclude("missing review rationale")
                continue

            if len(rationale.split()) < minimum_rationale_words:
                exclude("rationale too short; human re-review required")
                continue

            if not isinstance(reviewer, str) or not reviewer.strip():
                exclude("missing reviewer identity")
                continue

            if not isinstance(reviewed_at, str) or not _valid_timestamp(
                reviewed_at
            ):
                exclude("missing or invalid timezone-aware review timestamp")
                continue

            features = extract_feature_vector(text)

            examples.append(
                HumanLabelledExample(
                    sentence_id=sentence_id,
                    text=text,
                    label=label,
                    rationale=rationale,
                    reviewer=reviewer,
                    reviewed_at=reviewed_at,
                    features=tuple(features),
                )
            )

    label_counts = {
        level: sum(example.label == level for example in examples)
        for level in LEVELS
    }

    return {
        "source": str(source),
        "total_records": total,
        "eligible_records": len(examples),
        "excluded_records": len(excluded),
        "label_counts": label_counts,
        "examples": examples,
        "excluded": excluded,
    }
