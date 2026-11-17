import pytest

from vyrex_ai.data.english_level_classifier_v2_2 import (
    analyze_sentence_v2_2,
)


def test_v2_2_schema_and_feature_groups():
    result = analyze_sentence_v2_2("The researchers carefully evaluated the results.")

    assert result["schema_version"] == "2.2.0"
    assert set(result) == {
        "schema_version", "feature_status", "lexical",
        "syntactic", "grammatical", "frequency",
    }
    assert result["lexical"]["word_count"] == 6
    assert result["syntactic"]["sentence_count"] == 1


def test_empty_sentence_returns_safe_values():
    result = analyze_sentence_v2_2("")

    assert result["lexical"]["word_count"] == 0
    assert result["lexical"]["lexical_diversity"] == 0.0
    assert result["syntactic"]["sentence_count"] == 0
    assert result["frequency"]["frequency_coverage"] == 0.0


def test_repeated_words_affect_lexical_diversity():
    result = analyze_sentence_v2_2("The cat saw the cat.")

    assert result["lexical"]["word_count"] == 5
    assert result["lexical"]["unique_word_count"] == 3
    assert result["lexical"]["lexical_diversity"] == 0.6


def test_subordination_and_modal_detection():
    result = analyze_sentence_v2_2(
        "Although she could leave, she would stay."
    )

    assert result["syntactic"]["subordinate_marker_count"] == 1
    assert result["grammatical"]["modal_auxiliary_count"] == 2


def test_frequency_unavailable_is_not_fabricated(monkeypatch):
    from vyrex_ai.data import english_level_classifier_v2_2 as module

    monkeypatch.setattr(module, "_zipf_frequency", lambda word: None)

    result = module.analyze_sentence_v2_2(
        "The researchers evaluated the experiment."
    )

    assert result["feature_status"] == "frequency_unavailable"
    assert result["frequency"]["mean_zipf_frequency"] is None
    assert result["frequency"]["low_frequency_token_ratio"] is None
    assert result["frequency"]["frequency_coverage"] == 0.0


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        analyze_sentence_v2_2(None)
