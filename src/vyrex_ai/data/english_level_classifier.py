"""Transparent heuristic baseline for provisional English-level classification."""

from __future__ import annotations

import re
from collections import Counter

LEVELS = (
    "basic_english",
    "elementary_intermediate_english",
    "upper_intermediate_english",
    "advanced_english",
)

CONNECTORS = {
    "although", "however", "nevertheless", "whereas", "consequently",
    "therefore", "furthermore", "moreover", "despite", "unless",
    "whereby", "notwithstanding",
}

SUBORDINATORS = {
    "although", "because", "since", "unless", "whereas", "while",
    "though", "if", "when", "after", "before", "until", "once",
}

COMPLEX_WORDS = {
    "simultaneously", "sequentially", "consequently", "nevertheless",
    "short-sighted", "time-efficient", "time", "sequential",
    "approximately", "fundamental", "substantial", "interpretation",
    "sophisticated", "efficient", "efficiency", "circumstances",
    "particularly", "extraordinary", "environment", "methodology",
    "perspective", "significant", "demonstrate", "proportion",
}

INTERMEDIATE_WORDS = {
    "experience", "decision", "relationship", "opportunity",
    "development", "responsibility", "comparison", "improvement",
    "independent", "effective", "necessary", "environment",
}

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
SENTENCE_PATTERN = re.compile(r"[.!?;:]")


def _syllable_estimate(word: str) -> int:
    """Rough English syllable estimate; not a pronunciation dictionary."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)

    if word.endswith("e") and not word.endswith(("le", "ye")) and count > 1:
        count -= 1

    return max(1, count)


def analyze_sentence(text: str) -> dict:
    """Extract interpretable linguistic signals without external dependencies."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    words = WORD_PATTERN.findall(text.lower())
    if not words:
        raise ValueError("No English word tokens found")

    counts = Counter(words)
    word_count = len(words)
    unique_ratio = len(counts) / word_count
    average_length = sum(map(len, words)) / word_count
    long_words = sum(len(word) >= 9 for word in words)
    complex_words = sum(word in COMPLEX_WORDS for word in words)
    intermediate_words = sum(word in INTERMEDIATE_WORDS for word in words)
    connector_count = sum(word in CONNECTORS for word in words)
    subordinate_count = sum(word in SUBORDINATORS for word in words)
    syllables = sum(_syllable_estimate(word) for word in words)

    embedded_clause = bool(re.search(
        r"\b(?:idea|belief|claim|fact|assumption|thought)\s+that\b",
        text, re.I
    ))
    comparative_construction = bool(re.search(
        r"\bmore\b.{1,100}\bthan\b",
        text, re.I
    ))
    perfect_construction = bool(re.search(
        r"\b(?:has|have|had)\s+been\b",
        text, re.I
    ))
    hyphenated_complex = sum(
        word in {"short-sighted", "time-efficient"}
        for word in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)+", text.lower())
    )

    features = {
        "embedded_clause_marker": embedded_clause,
        "comparative_construction": comparative_construction,
        "perfect_construction": perfect_construction,
        "hyphenated_complex_expression_count": hyphenated_complex,
        "word_count": word_count,
        "unique_token_ratio": round(unique_ratio, 3),
        "average_word_length": round(average_length, 2),
        "long_word_count": long_words,
        "complex_vocabulary_hits": complex_words,
        "intermediate_vocabulary_hits": intermediate_words,
        "connector_count": connector_count,
        "subordinator_count": subordinate_count,
        "estimated_syllables_per_word": round(syllables / word_count, 2),
        "punctuation_count": len(SENTENCE_PATTERN.findall(text)),
        "has_relative_clause_marker": any(
            word in {"which", "whose", "whom", "where"} for word in words
        ),
        "has_conditional_marker": any(
            word in {"if", "unless"} for word in words
        ),
    }

    return features


def classify_sentence(text: str) -> dict:
    """Return a provisional level and transparent review-priority signals."""
    features = analyze_sentence(text)
    score = 0.0
    reasons = []

    if features["word_count"] >= 20:
        score += 1.0
        reasons.append("longer sentence structure")
    elif features["word_count"] >= 12:
        score += 0.5
        reasons.append("moderate sentence length")

    score += min(features["complex_vocabulary_hits"] * 0.6, 3.0)
    score += features["embedded_clause_marker"] * 0.8
    score += features["comparative_construction"] * 0.8
    score += features["perfect_construction"] * 0.2
    score += features["hyphenated_complex_expression_count"] * 0.3
    score += min(features["intermediate_vocabulary_hits"] * 0.5, 1.5)
    score += min(features["connector_count"] * 1.0, 2.0)
    score += min(features["subordinator_count"] * 0.6, 1.8)

    if features["embedded_clause_marker"]:
        reasons.append("embedded complement clause")
    if features["comparative_construction"]:
        reasons.append("comparative construction")
    if features["perfect_construction"]:
        reasons.append("perfect-tense construction")
    if features["hyphenated_complex_expression_count"]:
        reasons.append("hyphenated vocabulary")

    if features["has_relative_clause_marker"]:
        score += 0.6
        reasons.append("possible relative clause")

    if features["has_conditional_marker"]:
        score += 0.5
        reasons.append("conditional construction marker")

    if features["average_word_length"] >= 6.0:
        score += 0.5
        reasons.append("longer average word length")

    if features["estimated_syllables_per_word"] >= 2.0:
        score += 0.4
        reasons.append("higher estimated syllable density")

    if score < 0.8:
        level_index = 0
    elif score < 2.0:
        level_index = 1
    elif score < 3.8:
        level_index = 2
    else:
        level_index = 3

    # Review priority is rule-based, not a probability or accuracy estimate.
    review_reasons = []

    if features["word_count"] < 4:
        review_reasons.append("very short sentence")
    if features["word_count"] > 30:
        review_reasons.append("very long sentence")
    if features["word_count"] >= 4 and score == 0:
        review_reasons.append("limited linguistic evidence")
    if abs(score - 0.8) < 0.35:
        review_reasons.append("near basic/intermediate decision boundary")
    if abs(score - 2.0) < 0.35:
        review_reasons.append("near intermediate/upper-intermediate decision boundary")
    if abs(score - 3.8) < 0.35:
        review_reasons.append("near upper-intermediate/advanced decision boundary")

    review_required = bool(review_reasons)

    if not reasons:
        reasons.append("common vocabulary and limited structural signals")

    return {
        "predicted_level": LEVELS[level_index],
        "review_reasons": review_reasons,
        "confidence": None,
        "confidence_type": "not_calibrated",
        "review_required": review_required,
        "complexity_score": round(score, 3),
        "reasons": reasons,
        "features": features,
        "classification_status": "provisional_not_human_approved",
    }
