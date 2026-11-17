import pytest

from vyrex_ai.data.english_level_classifier_v2 import analyze_sentence


@pytest.mark.parametrize(
    ("sentence", "feature", "expected"),
    [
        ("I like apples.", "word_count", 3),
        ("Although it was raining, we continued walking.",
         "concessive_construction", True),
        ("Philosophers distinguish between a priori and a posteriori knowledge.",
         "academic_vocabulary_hits", 2),
        ("The book which I borrowed was interesting.",
         "relative_clause_marker", True),
        ("The question of which countries are prosperous remains.",
         "relative_clause_marker", False),
        ("The report was written by the researcher.",
         "passive_construction", True),
        ("The researcher wrote the report.",
         "passive_construction", False),
        ("Unless you study, you may fail.",
         "conditional_construction", True),
        ("She wants to learn English.",
         "infinitive_construction_count", 1),
        ("Having completed the work, she went home.",
         "participial_construction_count", 1),
        ("The book written by the professor was published.",
         "participial_construction_count", 1),
        ("The students are studying English.",
         "participial_construction_count", 0),
        ("We need to look into the problem.",
         "phrasal_verb_hits", 1),
    ],
)
def test_v2_linguistic_features(sentence, feature, expected):
    result = analyze_sentence(sentence)
    assert result[feature] == expected


@pytest.mark.parametrize("invalid_text", ["", "   ", None, 123])
def test_v2_rejects_invalid_input(invalid_text):
    with pytest.raises(ValueError):
        analyze_sentence(invalid_text)

def test_content_clause_is_not_misclassified_as_relative_clause():
    result = analyze_sentence(
        "The idea that reading makes you short-sighted is popular."
    )
    assert result["relative_clause_marker"] is False
    assert result["embedded_clause_marker"] is True


def test_noun_complement_that_clause_is_detected():
    result = analyze_sentence(
        "The claim that the policy will succeed is controversial."
    )
    assert result["embedded_clause_marker"] is True


def test_genuine_that_relative_clause_is_preserved():
    result = analyze_sentence(
        "The book that I borrowed was interesting."
    )
    assert result["relative_clause_marker"] is True


def test_reporting_verb_content_clause_is_not_relative():
    result = analyze_sentence(
        "The researchers have demonstrated that the results are significant."
    )
    assert result["embedded_clause_marker"] is True
    assert result["relative_clause_marker"] is False


def test_to_which_relative_clause_is_detected():
    result = analyze_sentence(
        "The extent to which socioeconomic circumstances influence "
        "educational attainment remains a subject of considerable debate."
    )
    assert result["relative_clause_marker"] is True
    assert result["embedded_clause_marker"] is False


def test_that_relative_clause_remains_detected():
    result = analyze_sentence(
        "The book that I borrowed was interesting."
    )
    assert result["relative_clause_marker"] is True
    assert result["embedded_clause_marker"] is False
from vyrex_ai.data.english_level_classifier_v2 import analyze_sentence


import pytest


@pytest.mark.parametrize(
    "sentence",
    [
        "She has completed the task.",
        "They have finished their work.",
        "He had written the report.",
        "The results have been published.",
    ],
)
def test_perfect_construction_is_detected(sentence):
    assert analyze_sentence(sentence)["perfect_construction"] is True


def test_simple_past_is_not_perfect_construction():
    assert analyze_sentence(
        "She completed the task yesterday."
    )["perfect_construction"] is False
