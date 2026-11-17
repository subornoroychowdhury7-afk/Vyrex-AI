"""Evaluation metrics for VYREX English-level classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate classification metrics without fitting a model."""

    if isinstance(y_true, (str, bytes)) or not isinstance(y_true, Sequence):
        raise TypeError("y_true must be a sequence of labels")

    if isinstance(y_pred, (str, bytes)) or not isinstance(y_pred, Sequence):
        raise TypeError("y_pred must be a sequence of labels")

    if not y_true:
        raise ValueError("Evaluation data cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")

    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in values
        ):
            raise ValueError(f"{name} must contain non-empty string labels")

    observed = set(y_true) | set(y_pred)

    if labels is None:
        label_order = sorted(observed)
    else:
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
            raise TypeError("labels must be a sequence of label names")

        if not labels:
            raise ValueError("labels cannot be empty")

        if any(
            not isinstance(label, str) or not label.strip()
            for label in labels
        ):
            raise ValueError("labels must contain non-empty strings")

        if len(set(labels)) != len(labels):
            raise ValueError("labels must not contain duplicates")

        label_order = list(labels)

        if not observed.issubset(set(label_order)):
            raise ValueError("labels must include every observed label")

    if len(set(y_true)) < 2:
        raise ValueError("Evaluation requires at least two true classes")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_order)
    }

    return {
        "sample_count": len(y_true),
        "labels": label_order,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=label_order,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=label_order,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=label_order,
        ).tolist(),
    }


def compare_prediction_sets(
    y_true: Sequence[str],
    predictions: Mapping[str, Sequence[str]],
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple classifiers against the same true labels."""

    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("predictions must be a non-empty mapping")

    results = {}

    for model_name, y_pred in predictions.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model names must be non-empty strings")

        results[model_name] = evaluate_predictions(
            y_true,
            y_pred,
            labels=labels,
        )

    return results
