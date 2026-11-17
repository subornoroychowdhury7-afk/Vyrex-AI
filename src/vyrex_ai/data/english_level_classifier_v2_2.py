"""Experimental VYREX AI V2.2 linguistic feature extraction.

Independent from V2/V2.1 scoring and ML schemas.
No proficiency labels or thresholds are produced.
Frequency features require the optional wordfreq package.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from vyrex_ai.data.english_level_classifier_v2 import WORD_PATTERN


FEATURE_SCHEMA_VERSION = "2.2.0"

WORDS = WORD_PATTERN
SENTENCE_PATTERN = re.compile(r"[.!?]+")
LONG_WORD_LENGTH = 7
RARE_WORD_ZIPF_THRESHOLD = 3.5

MODALS = {
    "can", "could", "may", "might", "must",
    "shall", "should", "will", "would",
}
AUXILIARIES = {
    "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did",
    *MODALS,
}
COORDINATORS = {"and", "but", "or", "nor", "yet", "so"}
SUBORDINATORS = {
    "although", "because", "before", "if", "once",
    "since", "though", "unless", "until", "when",
    "whenever", "whereas", "wherever", "while",
}
RELATIVE_MARKERS = {"who", "whom", "whose", "which", "that"}
PREPOSITIONS = {
    "about", "above", "across", "after", "against", "among",
    "around", "at", "before", "behind", "below", "beneath",
    "beside", "between", "beyond", "by", "despite", "during",
    "for", "from", "in", "inside", "into", "near", "of",
    "off", "on", "onto", "over", "through", "throughout",
    "to", "toward", "under", "upon", "with", "within", "without",
}
NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity",
    "ance", "ence", "ism",
)
NOMINALIZATION_EXCLUSIONS = {"experiment"}

PASSIVE_PATTERN = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+"
    r"(?:[a-z]+ly\s+){0,2}(?:[a-z]+ed|known|given|made|written|built|taken|"
    r"seen|shown|found|held|done|told|sent|bought)\b",
    re.I,
)
INFINITIVE_PATTERN = re.compile(r"\bto\s+[a-z]+\b", re.I)


@lru_cache(maxsize=32768)
def _zipf_frequency(word: str) -> float | None:
    """Return Zipf frequency, or None when wordfreq is unavailable."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None
    return float(zipf_frequency(word, "en"))


def _frequency_features(words: list[str]) -> dict[str, float | int | None]:
    if not words:
        return {
            "mean_zipf_frequency": None,
            "low_frequency_token_ratio": None,
            "low_frequency_type_ratio": None,
            "frequency_coverage": 0.0,
        }

    values = [_zipf_frequency(word) for word in words]
    known = [value for value in values if value is not None]

    if not known:
        return {
            "mean_zipf_frequency": None,
            "low_frequency_token_ratio": None,
            "low_frequency_type_ratio": None,
            "frequency_coverage": 0.0,
        }

    unique_words = set(words)
    unique_values = [_zipf_frequency(word) for word in unique_words]
    known_types = [value for value in unique_values if value is not None]

    return {
        "mean_zipf_frequency": round(sum(known) / len(known), 6),
        "low_frequency_token_ratio": round(
            sum(value < RARE_WORD_ZIPF_THRESHOLD for value in known)
            / len(known),
            6,
        ),
        "low_frequency_type_ratio": (
            round(
                sum(value < RARE_WORD_ZIPF_THRESHOLD for value in known_types)
                / len(known_types),
                6,
            )
            if known_types else None
        ),
        "frequency_coverage": round(len(known) / len(words), 6),
    }


def analyze_sentence_v2_2(text: str) -> dict:
    """Extract independent lexical, syntactic and grammatical diagnostics."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    words = [word.lower() for word in WORDS.findall(text)]
    counts = Counter(words)
    word_count = len(words)
    unique_count = len(counts)
    sentence_count = (
        len([part for part in SENTENCE_PATTERN.split(text) if part.strip()])
        if text.strip() else 0
    )

    modal_count = sum(counts[word] for word in MODALS)
    auxiliary_count = sum(counts[word] for word in AUXILIARIES)
    coordinator_count = sum(counts[word] for word in COORDINATORS)
    subordinator_count = sum(counts[word] for word in SUBORDINATORS)
    relative_count = sum(counts[word] for word in RELATIVE_MARKERS)
    preposition_marker_count = sum(counts[word] for word in PREPOSITIONS)

    nominalization_candidates = sum(
        word.endswith(NOMINALIZATION_SUFFIXES)
        and word not in NOMINALIZATION_EXCLUSIONS
        for word in words
    )

    lexical = {
        "word_count": word_count,
        "unique_word_count": unique_count,
        "lexical_diversity": (
            round(unique_count / word_count, 6) if word_count else 0.0
        ),
        "hapax_token_ratio": (
            round(sum(value == 1 for value in counts.values()) / word_count, 6)
            if word_count else 0.0
        ),
        "average_word_length": (
            round(sum(map(len, words)) / word_count, 6) if word_count else 0.0
        ),
        "long_word_ratio": (
            round(sum(len(word) >= LONG_WORD_LENGTH for word in words)
                  / word_count, 6)
            if word_count else 0.0
        ),
    }

    syntax = {
        "sentence_count": sentence_count,
        "coordinating_conjunction_count": coordinator_count,
        "subordinate_marker_count": subordinator_count,
        "relative_marker_count": relative_count,
        "preposition_marker_count": preposition_marker_count,
        "infinitive_marker_count": len(INFINITIVE_PATTERN.findall(text)),
        "nominalization_candidate_count": nominalization_candidates,
        "clause_marker_count_per_sentence": (
            round((subordinator_count + coordinator_count)
                  / sentence_count, 6)
            if sentence_count else 0.0
        ),
    }

    grammar = {
        "modal_auxiliary_count": modal_count,
        "auxiliary_count": auxiliary_count,
        "passive_construction_count": len(PASSIVE_PATTERN.findall(text)),
    }

    frequency = _frequency_features(words)
    coverage = frequency["frequency_coverage"]

    if not words:
        feature_status = "not_applicable"
    elif coverage == 0.0:
        feature_status = "frequency_unavailable"
    elif coverage < 1.0:
        feature_status = "frequency_partial"
    else:
        feature_status = "frequency_available"

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_status": feature_status,
        "lexical": lexical,
        "syntactic": syntax,
        "grammatical": grammar,
        "frequency": frequency,
    }
