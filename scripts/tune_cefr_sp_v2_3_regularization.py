from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vyrex_ai.data.english_level_cefr_sp_v2_3 import (
    SOURCE_HASHES,
    CEFR_LEVELS,
    _read_split,
    _feature_vectors,
    _make_examples,
)
from vyrex_ai.data.english_level_ml_models import create_models
from vyrex_ai.data.english_level_ml_split import (
    build_group_ids,
    create_stratified_group_splits,
)
from vyrex_ai.data.english_level_ml_evaluation import evaluate_predictions

DATA = ROOT / "data" / "processed" / "cefr_sp_wikiauto_v1"
OUTPUT = ROOT / "data" / "processed" / "cefr_sp_v2_3_regularization_v1.json"

C_VALUES = (0.1, 1.0, 10.0)

SELECTIONS = {
    "annotator_a": ("redundancy_reduced", "linear_svm"),
    "annotator_b": ("redundancy_reduced", "linear_svm"),
    "consensus": ("full", "logistic_regression"),
}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Report already exists: {OUTPUT}")

    experiment_path = DATA.parent / "cefr_sp_v2_3_experiment_v1.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))

    if experiment.get("experiment_status") != "completed":
        raise ValueError("Frozen baseline experiment is not completed")

    split_metadata = experiment.get("dataset", {}).get("splits", {})
    records = {}

    for split in ("train", "validation"):
        path = DATA / f"{split}.jsonl"
        expected_hash = SOURCE_HASHES[split]
        recorded_hash = split_metadata.get(split, {}).get("source_file_sha256")

        if recorded_hash != expected_hash:
            raise ValueError(f"Frozen source provenance mismatch: {split}")

        records[split], _ = _read_split(path, split)

    train_hashes = {r.text_hash for r in records["train"]}
    validation_hashes = {r.text_hash for r in records["validation"]}
    if train_hashes & validation_hashes:
        raise ValueError("Train/validation text leakage detected")

    cache = {}
    for split_records in records.values():
        for record in split_records:
            if record.text_hash not in cache:
                cache[record.text_hash] = _feature_vectors(record.text)

    results = {}

    for track, (configuration, model_name) in SELECTIONS.items():
        examples = _make_examples(
            records["train"], track, cache, configuration
        )
        validation = _make_examples(
            records["validation"], track, cache, configuration
        )

        groups = build_group_ids(examples)
        splits = create_stratified_group_splits(
            examples, n_splits=5, random_state=42
        )

        labels = [e.label for e in examples]
        features = [list(e.features) for e in examples]
        validation_x = [list(e.features) for e in validation]
        validation_y = [e.label for e in validation]

        baseline = create_models(random_state=42)[model_name]
        candidates = {}

        for c_value in C_VALUES:
            model_template = clone(baseline).set_params(
                classifier__C=c_value
            )
            oof = [None] * len(examples)

            for train_idx, val_idx in splits:
                train_groups = {groups[i] for i in train_idx}
                val_groups = {groups[i] for i in val_idx}
                if train_groups & val_groups:
                    raise RuntimeError("Grouped CV leakage detected")

                model = clone(model_template)
                model.fit(
                    [features[i] for i in train_idx],
                    [labels[i] for i in train_idx],
                )
                predictions = model.predict(
                    [features[i] for i in val_idx]
                )
                for index, prediction in zip(val_idx, predictions):
                    oof[index] = str(prediction)

            if any(value is None for value in oof):
                raise RuntimeError("Incomplete out-of-fold predictions")

            cv_metrics = evaluate_predictions(
                labels, oof, labels=CEFR_LEVELS
            )

            fitted = clone(model_template)
            fitted.fit(features, labels)
            validation_predictions = [
                str(value) for value in fitted.predict(validation_x)
            ]
            validation_metrics = evaluate_predictions(
                validation_y,
                validation_predictions,
                labels=CEFR_LEVELS,
            )

            candidates[str(c_value)] = {
                "cross_validation": cv_metrics,
                "validation": validation_metrics,
            }

        results[track] = {
            "configuration": configuration,
            "model": model_name,
            "feature_count": len(examples[0].features),
            "training_count": len(examples),
            "validation_count": len(validation),
            "candidates": candidates,
        }

        print(f"COMPLETED: {track}")

    report = {
        "schema_version": "1.0.0",
        "status": "completed",
        "experiment": "CEFR-SP V2.3 regularization tuning",
        "c_values": list(C_VALUES),
        "folds": 5,
        "seed": 42,
        "feature_extractions": len(cache),
        "validation_used_for_training": False,
        "test_split_accessed": False,
        "test_evaluation": "not_performed",
        "results": results,
        "production_approved": False,
        "confidence_calibrated": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("\nREGULARIZATION EXPERIMENT COMPLETED")
    print(f"Report: {OUTPUT.relative_to(ROOT)}")
    print(f"Feature extractions: {len(cache)}")

    for track, result in results.items():
        print(f"\n{track}:")
        for c_value, metrics in result["candidates"].items():
            cv = metrics["cross_validation"]
            val = metrics["validation"]
            print(
                f"C={c_value}: "
                f"CV macro-F1={cv['macro_f1']:.4f}, "
                f"validation macro-F1={val['macro_f1']:.4f}, "
                f"validation balanced accuracy="
                f"{val['balanced_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()

