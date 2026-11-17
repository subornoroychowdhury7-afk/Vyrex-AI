"""Leakage-aware V2.3 predictive feature ablation validation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vyrex_ai.data.english_level_classifier_v2_2 import (
    analyze_sentence_v2_2,
)
from vyrex_ai.data.english_level_ml_cv import cross_validate_models
from vyrex_ai.data.english_level_v2_2_ablation import (
    FEATURE_COUNTS,
    FEATURE_GROUPS,
    REMOVED_FEATURES,
    apply_v2_2_ablation,
)


@dataclass(frozen=True)
class _ValidationExample:
    """V2.3 feature vector with duplicate-safe text grouping."""

    text: str
    label: str
    features: tuple[float, ...]


def _feature_vector(
    features: dict[str, Any],
    configuration: str,
) -> tuple[float, ...]:
    ablated = apply_v2_2_ablation(features, configuration)
    vector = []

    for group in FEATURE_GROUPS:
        for name, value in ablated[group].items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"Unavailable or invalid feature: {group}.{name}"
                )

            vector.append(float(value))

    expected = FEATURE_COUNTS[configuration]

    if len(vector) != expected:
        raise ValueError(
            f"{configuration}: expected {expected} features, "
            f"got {len(vector)}"
        )

    return tuple(vector)


def run_v2_3_feature_validation(
    examples: Sequence,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare V2.2 feature configurations using approved human labels."""

    if isinstance(examples, (str, bytes)) or not isinstance(
        examples, Sequence
    ):
        raise TypeError("examples must be a sequence")

    if not examples:
        raise ValueError("Human-labelled examples are required")

    seen_ids = set()
    prepared = {name: [] for name in REMOVED_FEATURES}
    labels = []

    for index, example in enumerate(examples):
        sentence_id = getattr(example, "sentence_id", None)
        text = getattr(example, "text", None)
        label = getattr(example, "label", None)

        if not isinstance(sentence_id, str) or not sentence_id.strip():
            raise ValueError(f"Invalid sentence_id at example {index}")

        if sentence_id in seen_ids:
            raise ValueError(f"Duplicate sentence_id: {sentence_id}")

        seen_ids.add(sentence_id)

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text at example {index}")

        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid human label at example {index}")

        for field in ("rationale", "reviewer", "reviewed_at"):
            value = getattr(example, field, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Missing human-review provenance '{field}' "
                    f"at example {index}"
                )

        features = analyze_sentence_v2_2(text)

        for configuration in REMOVED_FEATURES:
            prepared[configuration].append(
                _ValidationExample(
                    text=text,
                    label=label,
                    features=_feature_vector(features, configuration),
                )
            )

        labels.append(label)

    results = {}

    for configuration, configuration_examples in prepared.items():
        evaluation = cross_validate_models(
            configuration_examples,
            n_splits=n_splits,
            random_state=random_state,
        )

        results[configuration] = {
            "feature_count": FEATURE_COUNTS[configuration],
            "removed_features": sorted(
                REMOVED_FEATURES[configuration]
            ),
            "metrics": evaluation["metrics"],
            "fold_reports": evaluation["fold_reports"],
            "evaluation_method": evaluation["evaluation_method"],
            "duplicate_text_leakage_checked": (
                evaluation["duplicate_text_leakage_checked"]
            ),
        }

    return {
        "schema_version": "2.3.0",
        "status": "completed",
        "sample_count": len(examples),
        "human_label_counts": dict(Counter(labels)),
        "configurations": results,
        "models": [
            "logistic_regression",
            "linear_svm",
            "random_forest",
        ],
        "evaluation_method": "stratified_group_cross_validation",
        "human_approved_labels_required": True,
        "production_approval": False,
    }
