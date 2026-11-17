from dataclasses import dataclass

import pytest

from vyrex_ai.data.english_level_ml_split import (
    build_group_ids,
    create_stratified_group_splits,
    normalize_sentence,
    sentence_group_id,
)


@dataclass
class Example:
    text: str
    label: str


def make_examples():
    examples = []

    for label in ("basic", "intermediate", "upper", "advanced"):
        for index in range(10):
            examples.append(
                Example(
                    text=f"{label} unique sentence number {index}",
                    label=label,
                )
            )

    return examples


def test_normalization_handles_case_and_whitespace():
    assert normalize_sentence("  HELLO   World ") == "hello world"


def test_unicode_normalization():
    assert sentence_group_id("café") == sentence_group_id("cafe\u0301")


def test_empty_sentence_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        sentence_group_id("  ")


def test_identical_sentences_share_group_id():
    assert sentence_group_id("Hello world") == sentence_group_id(
        " HELLO   WORLD "
    )


def test_group_ids_match_example_count():
    examples = make_examples()
    assert len(build_group_ids(examples)) == len(examples)


def test_grouped_splits_are_created():
    splits = create_stratified_group_splits(
        make_examples(),
        n_splits=5,
    )
    assert len(splits) == 5


def test_no_group_leakage_between_folds():
    examples = make_examples()
    groups = build_group_ids(examples)

    for train_indices, validation_indices in create_stratified_group_splits(
        examples,
        n_splits=5,
    ):
        train_groups = {groups[i] for i in train_indices}
        validation_groups = {groups[i] for i in validation_indices}
        assert train_groups.isdisjoint(validation_groups)


def test_conflicting_duplicate_labels_are_rejected():
    examples = [
        Example("The same sentence.", "basic"),
        Example(" THE SAME SENTENCE. ", "advanced"),
    ]

    with pytest.raises(ValueError, match="conflicting labels"):
        create_stratified_group_splits(examples, n_splits=2)


def test_insufficient_class_groups_are_rejected():
    examples = [
        Example(f"Basic sentence {i}", "basic")
        for i in range(4)
    ]
    examples += [
        Example(f"Advanced sentence {i}", "advanced")
        for i in range(2)
    ]

    with pytest.raises(ValueError, match="Insufficient independent groups"):
        create_stratified_group_splits(examples, n_splits=3)


def test_splits_are_reproducible():
    examples = make_examples()

    first = create_stratified_group_splits(
        examples, n_splits=5, random_state=42
    )
    second = create_stratified_group_splits(
        examples, n_splits=5, random_state=42
    )

    assert first == second


@pytest.mark.parametrize("value", [True, 1.5, "5", 1])
def test_invalid_fold_count_is_rejected(value):
    with pytest.raises(ValueError, match="n_splits"):
        create_stratified_group_splits(make_examples(), n_splits=value)


def test_invalid_random_state_is_rejected():
    with pytest.raises(ValueError, match="random_state"):
        create_stratified_group_splits(
            make_examples(), random_state=-1
        )


def test_empty_examples_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        create_stratified_group_splits([])
