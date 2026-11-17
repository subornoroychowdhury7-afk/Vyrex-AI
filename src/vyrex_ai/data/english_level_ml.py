"""Machine-learning feature preparation for English-level classification."""

from __future__ import annotations

import math

from typing import Any

from vyrex_ai.data.english_level_classifier_v2 import analyze_sentence


FEATURE_NAMES = (
    "word_count",
    "unique_token_ratio",
    "average_word_length",
    "long_word_count",
    "academic_vocabulary_hits",
    "technical_vocabulary_hits",
    "intermediate_vocabulary_hits",
    "connector_count",
    "subordinator_count",
    "passive_construction",
    "relative_clause_marker",
    "embedded_clause_marker",
    "infinitive_construction_count",
    "participial_construction_count",
    "conditional_construction",
    "concessive_construction",
    "comparative_construction",
    "perfect_construction",
    "phrasal_verb_hits",
    "punctuation_count",
)

BOOLEAN_FEATURES = frozenset({
    "passive_construction",
    "relative_clause_marker",
    "embedded_clause_marker",
    "conditional_construction",
    "concessive_construction",
    "comparative_construction",
    "perfect_construction",
})


def validate_features(features: dict[str, Any]) -> None:
    """Validate the complete V2 feature schema."""

    if not isinstance(features, dict):
        raise TypeError("features must be a dictionary")

    missing = set(FEATURE_NAMES) - features.keys()
    extra = features.keys() - set(FEATURE_NAMES)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if extra:
        raise ValueError(f"Unexpected features: {sorted(extra)}")

    for name in FEATURE_NAMES:
        value = features[name]

        if name in BOOLEAN_FEATURES:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{name} must be numeric")

        if name not in BOOLEAN_FEATURES:
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

            if value < 0:
                raise ValueError(f"{name} cannot be negative")

    if features["word_count"] < 1:
        raise ValueError("word_count must be positive")

    if not 0 <= features["unique_token_ratio"] <= 1:
        raise ValueError("unique_token_ratio must be between 0 and 1")


def features_to_vector(features: dict[str, Any]) -> list[float]:
    """Convert linguistic features into a stable numerical vector."""

    validate_features(features)

    return [
        float(features[name])
        for name in FEATURE_NAMES
    ]


def extract_feature_vector(text: str) -> list[float]:
    """Extract V2 linguistic features and convert them into a vector."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    features = analyze_sentence(text)
    return features_to_vector(features)
