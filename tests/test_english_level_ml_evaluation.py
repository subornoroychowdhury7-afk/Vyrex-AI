import pytest

from vyrex_ai.data.english_level_ml_evaluation import (
    compare_prediction_sets,
    evaluate_predictions,
)


def test_perfect_predictions():
    labels = ["basic", "intermediate", "advanced"]
    result = evaluate_predictions(labels, labels)

    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["weighted_f1"] == 1.0


def test_incorrect_predictions():
    result = evaluate_predictions(
        ["basic", "basic", "advanced", "advanced"],
        ["basic", "advanced", "advanced", "basic"],
    )

    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["sample_count"] == 4


def test_macro_f1_hand_calculated():
    result = evaluate_predictions(
        ["basic", "basic", "advanced"],
        ["basic", "advanced", "advanced"],
    )

    assert result["per_class"]["basic"]["f1"] == pytest.approx(2 / 3)
    assert result["per_class"]["advanced"]["f1"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)


def test_missing_class_is_preserved():
    labels = ["basic", "intermediate", "upper", "advanced"]

    result = evaluate_predictions(
        ["basic", "basic", "advanced", "advanced"],
        ["basic", "advanced", "advanced", "advanced"],
        labels=labels,
    )

    assert result["labels"] == labels
    assert "intermediate" in result["per_class"]
    assert result["per_class"]["intermediate"]["support"] == 0
    assert len(result["confusion_matrix"]) == 4


def test_confusion_matrix_order():
    result = evaluate_predictions(
        ["advanced", "basic", "advanced", "basic"],
        ["advanced", "advanced", "basic", "basic"],
        labels=["basic", "advanced"],
    )

    assert result["confusion_matrix"] == [
        [1, 1],
        [1, 1],
    ]


def test_per_class_support():
    result = evaluate_predictions(
        ["basic", "basic", "advanced"],
        ["basic", "advanced", "advanced"],
    )

    assert result["per_class"]["basic"]["support"] == 2
    assert result["per_class"]["advanced"]["support"] == 1


def test_empty_evaluation_data_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_predictions([], [])


def test_unequal_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_predictions(["basic"], ["basic", "advanced"])


def test_invalid_labels_are_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        evaluate_predictions(["basic", ""], ["basic", "advanced"])


def test_only_one_true_class_is_rejected():
    with pytest.raises(ValueError, match="at least two true classes"):
        evaluate_predictions(
            ["basic", "basic"],
            ["basic", "advanced"],
        )


def test_unlisted_prediction_label_is_rejected():
    with pytest.raises(ValueError, match="include every observed label"):
        evaluate_predictions(
            ["basic", "advanced"],
            ["basic", "unknown"],
            labels=["basic", "advanced"],
        )


def test_model_comparison():
    result = compare_prediction_sets(
        ["basic", "advanced", "basic", "advanced"],
        {
            "model_a": ["basic", "advanced", "basic", "advanced"],
            "model_b": ["basic", "basic", "advanced", "advanced"],
        },
    )

    assert set(result) == {"model_a", "model_b"}
    assert result["model_a"]["accuracy"] == 1.0
    assert result["model_b"]["accuracy"] == 0.5


def test_empty_model_comparison_is_rejected():
    with pytest.raises(ValueError, match="non-empty mapping"):
        compare_prediction_sets(["basic", "advanced"], {})


def test_model_comparison_rejects_empty_name():
    with pytest.raises(ValueError, match="Model names"):
        compare_prediction_sets(
            ["basic", "advanced"],
            {"": ["basic", "advanced"]},
        )
