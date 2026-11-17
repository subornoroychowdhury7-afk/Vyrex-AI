"""Provisional, interpretable V2 English-level scoring engine."""

from __future__ import annotations

from vyrex_ai.data.english_level_classifier_v2 import (
    LEVELS,
    analyze_sentence,
)


def score_features(features: dict) -> dict:
    """Calculate bounded linguistic complexity scores from V2 features."""

    required = {
        "word_count",
        "academic_vocabulary_hits",
        "technical_vocabulary_hits",
        "intermediate_vocabulary_hits",
        "embedded_clause_marker",
        "relative_clause_marker",
        "passive_construction",
        "infinitive_construction_count",
        "participial_construction_count",
        "perfect_construction",
        "comparative_construction",
        "connector_count",
        "subordinator_count",
        "conditional_construction",
        "concessive_construction",
        "average_word_length",
        "unique_token_ratio",
        "punctuation_count",
        "phrasal_verb_hits",
    }

    missing = required - features.keys()
    if missing:
        raise ValueError(
            f"Missing linguistic features: {sorted(missing)}"
        )

    if (
        isinstance(features["word_count"], bool)
        or not isinstance(features["word_count"], int)
        or features["word_count"] < 1
    ):
        raise ValueError("word_count must be a positive integer")

    def bounded(value, maximum):
        return min(max(value, 0), maximum)

    # 1. Vocabulary: maximum 30 points.
    academic = bounded(features["academic_vocabulary_hits"], 3) * 5
    technical = bounded(features["technical_vocabulary_hits"], 2) * 2.5
    intermediate = bounded(
        features["intermediate_vocabulary_hits"], 4
    ) * 2.5

    vocabulary_score = academic + technical + intermediate

    # 2. Grammar: maximum 35 points.
    grammar_score = (
        int(bool(features["embedded_clause_marker"])) * 7
        + int(bool(features["relative_clause_marker"])) * 7
        + int(bool(features["passive_construction"])) * 4
        + bounded(features["infinitive_construction_count"], 2) * 3
        + bounded(features["participial_construction_count"], 2) * 4
        + int(bool(features["perfect_construction"])) * 3
        + int(bool(features["comparative_construction"])) * 4
    )
    grammar_score = min(grammar_score, 35)

    # 3. Discourse: maximum 20 points.
    discourse_score = (
        bounded(features["connector_count"], 3) * 2
        + bounded(features["subordinator_count"], 3) * 2
        + int(bool(features["conditional_construction"])) * 4
        + int(bool(features["concessive_construction"])) * 4
    )

    # 4. Sentence complexity: maximum 15 points.
    word_length_score = (
        bounded(features["word_count"] - 5, 25) * 0.32
    )
    average_length_score = (
        bounded(features["average_word_length"] - 4, 3) * 1.2
    )
    punctuation_score = (
        bounded(features["punctuation_count"], 3) * 0.5
    )
    phrasal_score = bounded(features["phrasal_verb_hits"], 2) * 0.95

    complexity_score = min(
        word_length_score
        + average_length_score
        + punctuation_score
        + phrasal_score,
        15,
    )

    dimensions = {
        "vocabulary": round(vocabulary_score, 2),
        "grammar": round(grammar_score, 2),
        "discourse": round(discourse_score, 2),
        "sentence_complexity": round(complexity_score, 2),
    }

    total = round(sum(dimensions.values()), 2)

    return {
        "complexity_score": total,
        "maximum_score": 100,
        "dimension_scores": dimensions,
    }


def classify_features(features: dict) -> dict:
    """Assign a provisional level and identify cases needing review."""

    scoring = score_features(features)
    score = scoring["complexity_score"]

    # Initial engineering thresholds, not calibrated CEFR boundaries.
    if score < 20:
        level_index = 0
    elif score < 40:
        level_index = 1
    elif score < 65:
        level_index = 2
    else:
        level_index = 3

    review_reasons = []

    if features["word_count"] < 4:
        review_reasons.append("very short sentence")

    if features["word_count"] > 30:
        review_reasons.append("very long sentence")

    if (
        features["technical_vocabulary_hits"] > 0
        and score <= 10
    ):
        review_reasons.append(
            "technical vocabulary may reflect subject knowledge"
        )

    if score <= 5:
        review_reasons.append("limited linguistic evidence")

    for boundary, description in [
        (20, "basic/elementary-intermediate boundary"),
        (40, "elementary-intermediate/upper-intermediate boundary"),
        (65, "upper-intermediate/advanced boundary"),
    ]:
        if abs(score - boundary) <= 5:
            review_reasons.append(
                f"near {description}"
            )

    dimensions = scoring["dimension_scores"]

    reasons = [
        f"{name.replace('_', ' ')} score: {value}"
        for name, value in dimensions.items()
        if value > 0
    ]

    if not reasons:
        reasons.append("limited vocabulary and structural signals")

    return {
        "predicted_level": LEVELS[level_index],
        "complexity_score": score,
        "dimension_scores": dimensions,
        "reasons": reasons,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "confidence": None,
        "confidence_type": "not_calibrated",
        "classification_status": "provisional_not_human_approved",
    }


def classify_sentence(text: str) -> dict:
    """Extract features and return a provisional English-level prediction."""
    features = analyze_sentence(text)
    result = classify_features(features)
    result["features"] = features
    return result
