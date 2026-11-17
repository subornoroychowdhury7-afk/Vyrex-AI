from copy import deepcopy

import pytest

from vyrex_ai.data.english_level_classifier_v2 import analyze_sentence
from vyrex_ai.data.english_level_scoring_v2 import (
    LEVELS,
    classify_features,
    classify_sentence,
    score_features,
)


def test_scores_are_bounded():
    features = analyze_sentence(
        "Although the evidence was limited, the researchers continued."
    )
    result = score_features(features)

    assert 0 <= result["complexity_score"] <= result["maximum_score"]
    assert result["maximum_score"] == 100


def test_short_sentence_requires_review():
    result = classify_sentence("I like apples.")
    assert result["review_required"] is True


def test_automatic_classification_is_never_human_approved():
    result = classify_sentence(
        "Although it was raining, we continued walking."
    )
    assert result["classification_status"] == "provisional_not_human_approved"
    assert result["confidence"] is None
    assert result["confidence_type"] == "not_calibrated"


def test_missing_features_are_rejected():
    features = analyze_sentence("The researchers examined the results.")
    features.pop("academic_vocabulary_hits")

    with pytest.raises(ValueError, match="Missing linguistic features"):
        score_features(features)


def test_classification_does_not_modify_input_features():
    features = analyze_sentence(
        "The researchers examined the results carefully."
    )
    original = deepcopy(features)

    classify_features(features)

    assert features == original


def test_all_predictions_belong_to_supported_levels():
    samples = [
        "I like apples.",
        "Although it was raining, we continued walking.",
        "The idea that reading makes you short-sighted has been popular.",
        "It is more time-efficient to do several tasks sequentially than attempt to do them simultaneously.",
    ]

    for sentence in samples:
        result = classify_sentence(sentence)
        assert result["predicted_level"] in LEVELS
