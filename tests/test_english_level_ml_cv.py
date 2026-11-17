from dataclasses import dataclass

import pytest

from vyrex_ai.data.english_level_ml_cv import cross_validate_models


@dataclass
class Example:
    text: str
    label: str
    features: tuple[float, ...]


def make_examples():
    examples = []

    class_values = {
        "basic": 0.0,
        "intermediate": 2.0,
        "upper": 4.0,
        "advanced": 6.0,
    }

    for label, base in class_values.items():
        for index in range(15):
            examples.append(
                Example(
                    text=f"{label} synthetic sentence number {index}",
                    label=label,
                    features=(
                        base + index * 0.01,
                        base + 0.5,
                        float(index),
                    ),
                )
            )

    return examples


def test_cross_validation_runs_all_three_models():
    result = cross_validate_models(make_examples(), n_splits=5)

    assert set(result["metrics"]) == {
        "logistic_regression",
        "linear_svm",
        "random_forest",
    }


def test_every_example_gets_one_prediction():
    examples = make_examples()
    result = cross_validate_models(examples, n_splits=5)

    for predictions in result["out_of_fold_predictions"].values():
        assert len(predictions) == len(examples)
        assert all(isinstance(value, str) for value in predictions)


def test_cross_validation_reports_correct_counts():
    result = cross_validate_models(make_examples(), n_splits=5)

    assert result["sample_count"] == 60
    assert result["fold_count"] == 5
    assert result["feature_count"] == 3


def test_cross_validation_reports_class_counts():
    result = cross_validate_models(make_examples(), n_splits=5)

    assert result["class_counts"] == {
        "basic": 15,
        "intermediate": 15,
        "upper": 15,
        "advanced": 15,
    }


def test_cross_validation_checks_duplicate_leakage():
    result = cross_validate_models(make_examples(), n_splits=5)

    assert result["duplicate_text_leakage_checked"] is True


def test_cross_validation_is_reproducible():
    examples = make_examples()

    first = cross_validate_models(
        examples, n_splits=5, random_state=42
    )
    second = cross_validate_models(
        examples, n_splits=5, random_state=42
    )

    assert (
        first["out_of_fold_predictions"]
        == second["out_of_fold_predictions"]
    )


def test_empty_examples_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        cross_validate_models([])


def test_invalid_feature_vector_is_rejected():
    examples = make_examples()
    examples[0].features = (float("nan"), 1.0)

    with pytest.raises(ValueError, match="Invalid numeric feature"):
        cross_validate_models(examples)


def test_inconsistent_feature_lengths_are_rejected():
    examples = make_examples()
    examples[0].features = (1.0,)

    with pytest.raises(ValueError, match="equal length"):
        cross_validate_models(examples)


def test_incomplete_training_class_is_rejected():
    examples = [
        Example(
            text=f"basic example {i}",
            label="basic",
            features=(float(i), 1.0),
        )
        for i in range(5)
    ]

    examples += [
        Example(
            text=f"advanced example {i}",
            label="advanced",
            features=(float(i + 10), 2.0),
        )
        for i in range(4)
    ]

    with pytest.raises(ValueError, match="Insufficient independent groups"):
        cross_validate_models(examples, n_splits=5)
