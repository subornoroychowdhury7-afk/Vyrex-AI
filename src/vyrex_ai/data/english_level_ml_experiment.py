"""Reproducible experiments for VYREX English-level classifiers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn

from vyrex_ai.data.english_level_classifier_v2 import LEVELS
from vyrex_ai.data.english_level_ml import FEATURE_NAMES
from vyrex_ai.data.english_level_ml_cv import cross_validate_models
from vyrex_ai.data.english_level_ml_data import load_human_labelled_data
from vyrex_ai.data.english_level_ml_split import build_group_ids


EXPERIMENT_SCHEMA_VERSION = 1

MODEL_CONFIGURATIONS = {
    "logistic_regression": {
        "scaler": "StandardScaler",
        "classifier": "LogisticRegression",
        "max_iter": 2000,
        "class_weight": "balanced",
    },
    "linear_svm": {
        "scaler": "StandardScaler",
        "classifier": "LinearSVC",
        "class_weight": "balanced",
    },
    "random_forest": {
        "classifier": "RandomForestClassifier",
        "n_estimators": 300,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Experiment report already exists: {path}")

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if path.exists():
            raise FileExistsError(
                f"Experiment report already exists: {path}"
            )

        os.replace(temporary_path, path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_ml_experiment(
    dataset_path: str | Path,
    report_path: str | Path,
    n_splits: int = 5,
    random_state: int = 42,
    minimum_rationale_words: int = 3,
) -> dict[str, Any]:
    """Run and persist a human-label-only ML comparison experiment."""

    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, int)
        or n_splits < 2
    ):
        raise ValueError("n_splits must be an integer of at least 2")

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")

    source = Path(dataset_path)
    destination = Path(report_path)

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.resolve() == destination.resolve():
        raise ValueError("Report path cannot overwrite the source dataset")

    dataset_hash = _sha256_file(source)
    loaded = load_human_labelled_data(
        source,
        minimum_rationale_words=minimum_rationale_words,
    )
    examples = loaded["examples"]

    if not examples:
        raise ValueError("No eligible human-approved examples found")

    class_counts = Counter(example.label for example in examples)
    group_ids = build_group_ids(examples)

    groups_per_class = {
        level: len({
            group_ids[index]
            for index, example in enumerate(examples)
            if example.label == level
        })
        for level in LEVELS
    }

    missing_classes = [
        level for level in LEVELS if class_counts[level] == 0
    ]

    if missing_classes:
        raise ValueError(
            "Experiment requires every English level. Missing: "
            + ", ".join(missing_classes)
        )

    insufficient_classes = {
        level: count
        for level, count in groups_per_class.items()
        if count < n_splits
    }

    if insufficient_classes:
        details = ", ".join(
            f"{level}={count}"
            for level, count in insufficient_classes.items()
        )
        raise ValueError(
            f"Insufficient independent sentence groups for {n_splits}-fold "
            f"cross-validation: {details}"
        )

    results = cross_validate_models(
        examples,
        n_splits=n_splits,
        random_state=random_state,
    )

    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "sha256": dataset_hash,
            "total_records": loaded["total_records"],
            "eligible_records": loaded["eligible_records"],
            "excluded_records": loaded["excluded_records"],
            "class_counts": dict(class_counts),
            "independent_groups_per_class": groups_per_class,
            "labels": list(LEVELS),
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "minimum_rationale_words": minimum_rationale_words,
        },
        "evaluation": {
            "method": "stratified_group_cross_validation",
            "folds": n_splits,
            "random_state": random_state,
            "duplicate_text_leakage_checked": True,
            "human_approved_labels_only": True,
            "confidence_calibration": "not_performed",
        },
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        },
        "models": MODEL_CONFIGURATIONS,
        "results": {
            "sample_count": results["sample_count"],
            "feature_count": results["feature_count"],
            "class_counts": results["class_counts"],
            "fold_count": results["fold_count"],
            "metrics": results["metrics"],
            "fold_reports": results["fold_reports"],
            "out_of_fold_predictions": results[
                "out_of_fold_predictions"
            ],
        },
        "interpretation": {
            "confidence_scores_calibrated": False,
            "production_model_approved": False,
            "note": (
                "Cross-validation results are experimental estimates. "
                "They do not establish production readiness or calibrated "
                "prediction confidence."
            ),
        },
    }

    _write_json_atomic(destination, report)

    return {
        "report_path": str(destination),
        "dataset_sha256": dataset_hash,
        "sample_count": results["sample_count"],
        "class_counts": dict(class_counts),
        "fold_count": results["fold_count"],
        "model_names": list(results["metrics"]),
        "status": "completed",
    }
