"""Isolated V2.2 feature ablation configurations.

Does not modify the V2.2 extractor or produce proficiency predictions.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ABLATION_SCHEMA_VERSION = "1.0.0"

FEATURE_GROUPS = (
    "lexical",
    "syntactic",
    "grammatical",
    "frequency",
)

FEATURE_COUNTS = {
    "full": 21,
    "constant_removed": 20,
    "redundancy_reduced": 17,
}

REMOVED_FEATURES = {
    "full": frozenset(),
    "constant_removed": frozenset({
        "frequency.frequency_coverage",
    }),
    "redundancy_reduced": frozenset({
        "frequency.frequency_coverage",
        "frequency.low_frequency_type_ratio",
        "lexical.hapax_token_ratio",
        "lexical.unique_word_count",
    }),
}


def _flatten_numeric_features(features: dict[str, Any]) -> dict[str, float | int | None]:
    flattened = {}

    for group in FEATURE_GROUPS:
        values = features.get(group)

        if not isinstance(values, dict):
            raise ValueError(f"Missing or invalid feature group: {group}")

        for name, value in values.items():
            if isinstance(value, bool):
                raise ValueError(f"Boolean feature is unsupported: {group}.{name}")

            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric feature: {group}.{name}")

            flattened[f"{group}.{name}"] = value

    return flattened


def apply_v2_2_ablation(
    features: dict[str, Any],
    configuration: str,
) -> dict[str, Any]:
    """Return a copied feature structure with selected features removed."""
    if configuration not in REMOVED_FEATURES:
        raise ValueError(f"Unknown ablation configuration: {configuration}")

    flattened = _flatten_numeric_features(features)
    removed = REMOVED_FEATURES[configuration]

    missing = removed.difference(flattened)
    if missing:
        raise ValueError(
            f"Required ablation features missing: {sorted(missing)}"
        )

    result = deepcopy(features)

    for feature_name in removed:
        group, name = feature_name.split(".", maxsplit=1)
        del result[group][name]

    actual_count = sum(
        len(result[group])
        for group in FEATURE_GROUPS
    )

    expected_count = FEATURE_COUNTS[configuration]

    if actual_count != expected_count:
        raise ValueError(
            f"Feature count mismatch: expected {expected_count}, "
            f"got {actual_count}"
        )

    return result


def compare_v2_2_ablation(
    features: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Generate all three isolated configurations."""
    return {
        name: apply_v2_2_ablation(features, name)
        for name in REMOVED_FEATURES
    }
