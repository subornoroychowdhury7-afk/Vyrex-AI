from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vyrex_ai.data.english_level_cefr_sp_v2_3 import (
    CEFR_LEVELS,
    _feature_vectors,
    _make_examples,
    _read_split,
    _sha256_file,
    _write_json_atomic,
)
from vyrex_ai.data.english_level_ml_evaluation import evaluate_predictions
from vyrex_ai.data.english_level_ml_models import create_models

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Report already exists: {args.report}")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "model_selection_frozen":
        raise ValueError("Model selection is not frozen")

    experiment_hash = _sha256_file(args.experiment)
    if experiment_hash != selection["experiment"]["sha256"]:
        raise ValueError("Experiment report hash mismatch")

    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    if experiment.get("experiment_status") != "completed":
        raise ValueError("Original experiment is not completed")

    train, train_meta = _read_split(
        args.dataset / "train.jsonl", "train"
    )
    validation, validation_meta = _read_split(
        args.dataset / "validation.jsonl", "validation"
    )
    test, test_meta = _read_split(
        args.dataset / "test.jsonl", "test"
    )

    for split, metadata in (
        ("train", train_meta),
        ("validation", validation_meta),
        ("test", test_meta),
    ):
        expected = selection["dataset"]["source_file_sha256"][split]
        if metadata["source_file_sha256"] != expected:
            raise ValueError(f"{split} source hash mismatch")

    train_hashes = {r.text_hash for r in train}
    validation_hashes = {r.text_hash for r in validation}
    test_hashes = {r.text_hash for r in test}

    if train_hashes & validation_hashes:
        raise ValueError("Train/validation text leakage detected")
    if train_hashes & test_hashes:
        raise ValueError("Train/test text leakage detected")
    if validation_hashes & test_hashes:
        raise ValueError("Validation/test text leakage detected")

    feature_cache = {}
    for record in train + test:
        if record.text_hash not in feature_cache:
            feature_cache[record.text_hash] = _feature_vectors(record.text)

    selections = selection["selected_models"]
    results = {}

    for track in ("annotator_a", "annotator_b", "consensus"):
        chosen = selections[track]
        configuration = chosen["configuration"]
        model_name = chosen["model"]

        training = _make_examples(
            train, track, feature_cache, configuration
        )
        testing = _make_examples(
            test, track, feature_cache, configuration
        )

        if not training or not testing:
            raise ValueError(f"Empty train/test data for {track}")

        if len({e.label for e in training}) != 6:
            raise ValueError(f"Training classes incomplete for {track}")

        if len({e.label for e in testing}) < 2:
            raise ValueError(f"Insufficient test classes for {track}")

        if len(training[0].features) != chosen["feature_count"]:
            raise ValueError(f"Feature count mismatch for {track}")

        model = create_models(args.seed)[model_name]
        x_train = [list(e.features) for e in training]
        y_train = [e.label for e in training]
        x_test = [list(e.features) for e in testing]
        y_test = [e.label for e in testing]

        model.fit(x_train, y_train)
        predictions = [str(p) for p in model.predict(x_test)]

        results[track] = {
            "selected_configuration": configuration,
            "selected_model": model_name,
            "feature_count": chosen["feature_count"],
            "training_records": len(training),
            "test_records": len(testing),
            "training_class_counts": dict(Counter(y_train)),
            "test_class_counts": dict(Counter(y_test)),
            "metrics": evaluate_predictions(
                y_test, predictions, labels=CEFR_LEVELS
            ),
        }

    report = {
        "schema_version": "1.0.0",
        "status": "completed",
        "evaluation_type": "one_time_frozen_model_held_out_test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": args.seed,
        "authorization": {
            "user_authorized_in_conversation": True,
            "selection_artifact_modified": False,
        },
        "selection": {
            "path": str(args.selection),
            "sha256": _sha256_file(args.selection),
        },
        "experiment": {
            "path": str(args.experiment),
            "sha256": experiment_hash,
        },
        "dataset": {
            "path": str(args.dataset),
            "source_hashes": {
                "train": train_meta["source_file_sha256"],
                "validation": validation_meta["source_file_sha256"],
                "test": test_meta["source_file_sha256"],
            },
            "jsonl_sha256": {
                "train": _sha256_file(args.dataset / "train.jsonl"),
                "validation": _sha256_file(args.dataset / "validation.jsonl"),
                "test": _sha256_file(args.dataset / "test.jsonl"),
            },
            "unique_records": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "cross_split_text_overlap": False,
        },
        "feature_extractions": len(feature_cache),
        "results": results,
        "limitations": [
            "Test results must not be used for model or threshold tuning.",
            "Human annotations are not established as a gold standard.",
            "Consensus excludes annotation disagreements.",
            "Confidence calibration has not been performed.",
            "Commercial use has not been approved.",
        ],
        "production_approved": False,
        "commercial_use_approved": False,
    }

    _write_json_atomic(args.report, report)
    print("HELD-OUT TEST EVALUATION COMPLETED")
    print(f"Report: {args.report}")
    print(f"Feature extractions: {len(feature_cache)}")
    for track, result in results.items():
        metrics = result["metrics"]
        print(
            f"{track}: model={result['selected_model']}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )

if __name__ == "__main__":
    main()
