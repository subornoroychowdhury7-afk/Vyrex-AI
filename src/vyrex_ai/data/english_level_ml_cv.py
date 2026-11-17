"""Leakage-aware cross-validation for VYREX English classifiers."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from vyrex_ai.data.english_level_ml_evaluation import (
    compare_prediction_sets,
)
from vyrex_ai.data.english_level_ml_models import create_models
from vyrex_ai.data.english_level_ml_split import (
    build_group_ids,
    create_stratified_group_splits,
)


def cross_validate_models(
    examples: Sequence,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train candidate models using duplicate-safe out-of-fold evaluation."""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("examples must be a sequence")

    if not examples:
        raise ValueError("At least one labelled example is required")

    labels = []
    features = []

    for index, example in enumerate(examples):
        label = getattr(example, "label", None)
        vector = getattr(example, "features", None)

        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid label at example {index}")

        if not isinstance(vector, (tuple, list)) or not vector:
            raise ValueError(f"Invalid feature vector at example {index}")

        for value in vector:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"Invalid numeric feature at example {index}"
                )

        labels.append(label)
        features.append([float(value) for value in vector])

    feature_lengths = {len(vector) for vector in features}

    if len(feature_lengths) != 1:
        raise ValueError("All feature vectors must have equal length")

    groups = build_group_ids(examples)

    splits = create_stratified_group_splits(
        examples,
        n_splits=n_splits,
        random_state=random_state,
    )

    all_classes = set(labels)
    model_names = (
        "logistic_regression",
        "linear_svm",
        "random_forest",
    )

    out_of_fold = {
        name: [None] * len(examples)
        for name in model_names
    }

    fold_reports = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splits,
        start=1,
    ):
        train_classes = {labels[i] for i in train_indices}

        if train_classes != all_classes:
            raise ValueError(
                f"Fold {fold_number} training data does not contain "
                "every class"
            )

        train_groups = {groups[i] for i in train_indices}
        validation_groups = {groups[i] for i in validation_indices}

        if train_groups & validation_groups:
            raise RuntimeError(
                f"Duplicate-text leakage in fold {fold_number}"
            )

        x_train = [features[i] for i in train_indices]
        y_train = [labels[i] for i in train_indices]
        x_validation = [features[i] for i in validation_indices]

        models = create_models(random_state=random_state)

        for name, model in models.items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_validation)

            for index, prediction in zip(
                validation_indices,
                predictions,
            ):
                if out_of_fold[name][index] is not None:
                    raise RuntimeError(
                        f"Example {index} received multiple predictions"
                    )

                out_of_fold[name][index] = str(prediction)

        fold_reports.append({
            "fold": fold_number,
            "training_count": len(train_indices),
            "validation_count": len(validation_indices),
            "training_class_counts": dict(
                Counter(y_train)
            ),
            "validation_class_counts": dict(
                Counter(labels[i] for i in validation_indices)
            ),
        })

    for name, predictions in out_of_fold.items():
        if any(prediction is None for prediction in predictions):
            raise RuntimeError(
                f"Some examples lack out-of-fold predictions for {name}"
            )

    predictions = {
        name: [str(value) for value in values]
        for name, values in out_of_fold.items()
    }

    metrics = compare_prediction_sets(
        labels,
        predictions,
        labels=sorted(all_classes),
    )

    return {
        "sample_count": len(examples),
        "class_counts": dict(Counter(labels)),
        "fold_count": len(splits),
        "feature_count": len(features[0]),
        "metrics": metrics,
        "out_of_fold_predictions": predictions,
        "fold_reports": fold_reports,
        "evaluation_method": "stratified_group_cross_validation",
        "duplicate_text_leakage_checked": True,
    }
