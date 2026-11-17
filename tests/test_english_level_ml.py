from copy import deepcopy
import math
import pytest

from vyrex_ai.data.english_level_ml import (
    BOOLEAN_FEATURES,
    FEATURE_NAMES,
    extract_feature_vector,
    features_to_vector,
    validate_features,
)
from vyrex_ai.data.english_level_classifier_v2 import analyze_sentence


SAMPLE = "Although it was raining, we continued walking."


def test_feature_schema_contains_20_features():
    assert len(FEATURE_NAMES) == 20
    assert len(set(FEATURE_NAMES)) == 20


def test_feature_vector_has_stable_length():
    assert len(extract_feature_vector(SAMPLE)) == 20


def test_feature_vector_contains_finite_numbers():
    vector = extract_feature_vector(SAMPLE)
    assert all(math.isfinite(value) for value in vector)


def test_boolean_features_are_converted_to_numbers():
    features = analyze_sentence(SAMPLE)
    vector = features_to_vector(features)
    for name in BOOLEAN_FEATURES:
        assert vector[FEATURE_NAMES.index(name)] in (0.0, 1.0)


def test_missing_feature_is_rejected():
    features = analyze_sentence(SAMPLE)
    features.pop("word_count")
    with pytest.raises(ValueError, match="Missing features"):
        validate_features(features)


def test_unexpected_feature_is_rejected():
    features = analyze_sentence(SAMPLE)
    features["unknown_feature"] = 1
    with pytest.raises(ValueError, match="Unexpected features"):
        validate_features(features)


def test_boolean_feature_rejects_integer():
    features = analyze_sentence(SAMPLE)
    features["passive_construction"] = 1
    with pytest.raises(ValueError, match="must be boolean"):
        validate_features(features)


def test_negative_numeric_feature_is_rejected():
    features = analyze_sentence(SAMPLE)
    features["word_count"] = -1
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_features(features)


def test_invalid_unique_token_ratio_is_rejected():
    features = analyze_sentence(SAMPLE)
    features["unique_token_ratio"] = 1.5
    with pytest.raises(ValueError, match="unique_token_ratio"):
        validate_features(features)


def test_empty_sentence_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        extract_feature_vector("")


def test_feature_conversion_does_not_mutate_input():
    features = analyze_sentence(SAMPLE)
    original = deepcopy(features)
    features_to_vector(features)
    assert features == original

def test_nan_numeric_feature_is_rejected():
    features = analyze_sentence(SAMPLE)
    features["average_word_length"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        validate_features(features)


def test_infinite_numeric_feature_is_rejected():
    features = analyze_sentence(SAMPLE)
    features["average_word_length"] = float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        validate_features(features)
