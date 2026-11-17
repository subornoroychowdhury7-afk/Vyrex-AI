import pytest

from vyrex_ai.data.english_level_classifier_v2_1 import (
    analyze_sentence_v2_1,
)


def test_adjective_is_not_counted_as_auxiliary_chain():
    result = analyze_sentence_v2_1(
        "Although the evidence was limited, researchers continued."
    )
    assert result["auxiliary_chain_count"] == 0


def test_prepositional_before_is_not_subordinate_clause():
    result = analyze_sentence_v2_1(
        "The system should have been evaluated before deployment."
    )
    assert result["subordinate_marker_count"] == 0


def test_plural_nominalization_is_detected():
    result = analyze_sentence_v2_1(
        "The researchers explained their conclusions."
    )
    assert result["nominalization_candidate_count"] >= 1


def test_complete_modal_perfect_passive_chain_counts_once():
    result = analyze_sentence_v2_1(
        "The system should have been evaluated."
    )
    assert result["auxiliary_chain_count"] == 1
from vyrex_ai.data.english_level_classifier_v2_1 import (
    analyze_sentence_v2_1,
)


def test_modal_auxiliary_is_detected():
    result = analyze_sentence_v2_1(
        "Students should complete their assignments."
    )
    assert result["modal_auxiliary_count"] == 1


def test_coordinating_conjunction_is_detected():
    result = analyze_sentence_v2_1(
        "The researchers studied the data and published results."
    )
    assert result["coordinating_conjunction_count"] == 1


def test_genuine_subordinate_clause_is_detected():
    result = analyze_sentence_v2_1(
        "Although the evidence was limited, researchers continued."
    )
    assert result["subordinate_marker_count"] == 1


def test_nominalization_singular_and_plural():
    result = analyze_sentence_v2_1(
        "The implementation produced several conclusions."
    )
    assert result["nominalization_candidate_count"] >= 2


def test_simple_sentence_has_no_auxiliary_chain():
    result = analyze_sentence_v2_1(
        "Researchers examine evidence carefully."
    )
    assert result["auxiliary_chain_count"] == 0


def test_multiple_sentences_are_counted():
    result = analyze_sentence_v2_1(
        "Researchers studied the evidence. They published results!"
    )
    assert result["sentence_count"] == 2


def test_feature_ratios_remain_in_valid_range():
    result = analyze_sentence_v2_1(
        "The researchers carefully evaluated the experimental results."
    )
    assert 0 <= result["lexical_diversity"] <= 1
    assert 0 <= result["long_word_ratio"] <= 1
    assert 0 <= result["preposition_ratio"] <= 1
    assert 0 <= result["nominalization_ratio"] <= 1


def test_empty_text_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        analyze_sentence_v2_1("")
def test_object_relative_clause_is_detected():
    result = analyze_sentence_v2_1(
        "The book that I purchased yesterday was expensive."
    )
    assert result["relative_clause_count"] == 1


def test_subject_relative_clause_is_detected():
    result = analyze_sentence_v2_1(
        "The researcher who works here published a paper."
    )
    assert result["relative_clause_count"] == 1


def test_content_clause_is_not_counted_as_relative_clause():
    result = analyze_sentence_v2_1(
        "Researchers believe that the system works."
    )
    assert result["relative_clause_count"] == 0


def test_experiment_is_not_nominalization():
    result = analyze_sentence_v2_1(
        "The experiment failed."
    )
    assert result["nominalization_candidate_count"] == 0


def test_plural_questions_are_not_nominalizations():
    result = analyze_sentence_v2_1(
        "Several important questions remain."
    )
    assert result["nominalization_candidate_count"] == 0


def test_genuine_nominalizations_remain_detected():
    result = analyze_sentence_v2_1(
        "The implementation led to important conclusions."
    )
    assert result["nominalization_candidate_count"] >= 2
def test_content_clause_after_idea_is_not_relative():
    result = analyze_sentence_v2_1(
        "The idea that reading makes you short-sighted is popular."
    )
    assert result["relative_clause_count"] == 0


def test_content_clause_after_believe_is_not_relative():
    result = analyze_sentence_v2_1(
        "I believe that the system works."
    )
    assert result["relative_clause_count"] == 0


def test_relative_clause_with_determiner_subject():
    result = analyze_sentence_v2_1(
        "The book that the student purchased was expensive."
    )
    assert result["relative_clause_count"] == 1


def test_relative_clause_with_preceding_preposition():
    result = analyze_sentence_v2_1(
        "The system in which the data is stored is secure."
    )
    assert result["relative_clause_count"] == 1


def test_relative_clause_with_who():
    result = analyze_sentence_v2_1(
        "The engineer who designed the system received an award."
    )
    assert result["relative_clause_count"] == 1

def test_demonstrative_that_is_not_relative_clause():
    result = analyze_sentence_v2_1(
        "And that settles it."
    )
    assert result["relative_clause_count"] == 0


def test_fused_relative_that_which_is_counted_once():
    result = analyze_sentence_v2_1(
        "Never attribute to malice that which is adequately explained by stupidity."
    )
    assert result["relative_clause_count"] == 1


def test_that_which_does_not_break_regular_relative_detection():
    result = analyze_sentence_v2_1(
        "The book that I purchased yesterday was expensive."
    )
    assert result["relative_clause_count"] == 1

def test_ordinary_nouns_are_not_nominalization_candidates():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        _count_nominalization_candidates,
    )

    words = ["city", "romance", "chance", "testament"]
    assert _count_nominalization_candidates(words) == 0


def test_specific_genuine_nominalization_candidates_remain_detected():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        _count_nominalization_candidates,
    )

    words = [
        "popularity",
        "similarities",
        "necessity",
        "complexity",
        "acceptance",
        "election",
        "detection",
        "inconvenience",
    ]
    assert _count_nominalization_candidates(words) == 8

def test_demonstrative_that_after_completed_clause_is_not_relative():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        analyze_sentence_v2_1,
    )

    sentence = (
        "Good people can do evil - that takes religion."
    )
    assert analyze_sentence_v2_1(sentence)["relative_clause_count"] == 0


def test_that_what_content_clause_is_not_relative():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        _count_relative_clauses,
    )

    words = ["realized", "that", "what", "you", "heard", "is", "not"]
    assert _count_relative_clauses(words) == 0


def test_thats_been_is_detected_as_perfect_auxiliary_chain():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        analyze_sentence_v2_1,
    )

    result = analyze_sentence_v2_1(
        "That's been devastating for fish."
    )
    assert result["auxiliary_count"] == 2
    assert result["auxiliary_chain_count"] == 1


def test_thats_already_been_is_detected_as_perfect_auxiliary_chain():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        analyze_sentence_v2_1,
    )

    result = analyze_sentence_v2_1(
        "That's already been completed."
    )
    assert result["auxiliary_count"] == 2
    assert result["auxiliary_chain_count"] == 1


def test_thats_running_is_detected_as_single_auxiliary():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        analyze_sentence_v2_1,
    )

    result = analyze_sentence_v2_1("That's running smoothly.")
    assert result["auxiliary_count"] == 1
    assert result["auxiliary_chain_count"] == 0


def test_thats_noun_phrase_is_not_detected_as_auxiliary():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        analyze_sentence_v2_1,
    )

    result = analyze_sentence_v2_1("That's a good idea.")
    assert result["auxiliary_count"] == 0
    assert result["auxiliary_chain_count"] == 0



def test_excluded_lexicalized_nouns_are_not_nominalization_candidates():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        _count_nominalization_candidates,
    )

    words = [
        "serendipity",
        "sucker",
        "inconvenience",
        "evolution",
    ]
    assert _count_nominalization_candidates(words) == 2


def test_genuine_nominalizations_remain_detected():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        _count_nominalization_candidates,
    )

    words = [
        "popularity",
        "similarities",
        "necessity",
        "complexity",
        "acceptance",
        "election",
        "detection",
        "humanity",
        "stupidity",
    ]
    assert _count_nominalization_candidates(words) == 9


def test_participial_adjectives_are_not_nominalization_candidates():
    from vyrex_ai.data.english_level_classifier_v2_1 import (
        _count_nominalization_candidates,
    )

    words = ["increased", "reduced", "improved"]
    assert _count_nominalization_candidates(words) == 0

