from copy import deepcopy

import pytest

from vyrex_ai.data.english_level_classifier_v2_2 import analyze_sentence_v2_2
from vyrex_ai.data.english_level_v2_2_ablation import (
    FEATURE_COUNTS,
    apply_v2_2_ablation,
    compare_v2_2_ablation,
)


@pytest.fixture
def features():
    return analyze_sentence_v2_2(
        "Although the researchers carefully evaluated the results, "
        "the development of the system was considered significant."
    )


def test_all_configurations_have_expected_counts(features):
    results = compare_v2_2_ablation(features)

    for name, result in results.items():
        count = sum(
            len(result[group])
            for group in ("lexical", "syntactic", "grammatical", "frequency")
        )
        assert count == FEATURE_COUNTS[name]


def test_full_configuration_preserves_all_features(features):
    assert apply_v2_2_ablation(features, "full") == features


def test_constant_removed_excludes_frequency_coverage(features):
    result = apply_v2_2_ablation(features, "constant_removed")

    assert "frequency_coverage" not in result["frequency"]
    assert "low_frequency_type_ratio" in result["frequency"]


def test_redundancy_reduced_excludes_selected_features(features):
    result = apply_v2_2_ablation(features, "redundancy_reduced")

    assert "frequency_coverage" not in result["frequency"]
    assert "low_frequency_type_ratio" not in result["frequency"]
    assert "hapax_token_ratio" not in result["lexical"]
    assert "unique_word_count" not in result["lexical"]


def test_original_features_are_not_mutated(features):
    original = deepcopy(features)

    compare_v2_2_ablation(features)

    assert features == original


def test_result_does_not_mutate_original_nested_groups(features):
    result = apply_v2_2_ablation(features, "redundancy_reduced")

    result["lexical"]["word_count"] = 999

    assert features["lexical"]["word_count"] != 999


def test_unknown_configuration_is_rejected(features):
    with pytest.raises(ValueError, match="Unknown ablation"):
        apply_v2_2_ablation(features, "unknown")


def test_missing_required_feature_is_rejected(features):
    modified = deepcopy(features)
    del modified["frequency"]["frequency_coverage"]

    with pytest.raises(ValueError, match="features missing"):
        apply_v2_2_ablation(modified, "constant_removed")


def test_missing_feature_group_is_rejected(features):
    modified = deepcopy(features)
    del modified["syntactic"]

    with pytest.raises(ValueError, match="feature group"):
        apply_v2_2_ablation(modified, "full")


def test_boolean_feature_is_rejected(features):
    modified = deepcopy(features)
    modified["lexical"]["word_count"] = True

    with pytest.raises(ValueError, match="Boolean feature"):
        apply_v2_2_ablation(modified, "full")


def test_non_numeric_feature_is_rejected(features):
    modified = deepcopy(features)
    modified["lexical"]["word_count"] = "invalid"

    with pytest.raises(ValueError, match="Non-numeric feature"):
        apply_v2_2_ablation(modified, "full")
