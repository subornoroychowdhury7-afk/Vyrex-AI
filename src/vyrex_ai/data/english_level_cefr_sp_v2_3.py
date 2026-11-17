"""Leakage-safe CEFR-SP V2.3 annotator-aware experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn

from vyrex_ai.data.english_level_classifier_v2_2 import (
    analyze_sentence_v2_2,
)
from vyrex_ai.data.english_level_ml_cv import cross_validate_models
from vyrex_ai.data.english_level_ml_evaluation import evaluate_predictions
from vyrex_ai.data.english_level_ml_models import create_models
from vyrex_ai.data.english_level_ml_split import (
    build_group_ids,
    sentence_group_id,
)
from vyrex_ai.data.english_level_v2_2_ablation import (
    FEATURE_COUNTS,
    FEATURE_GROUPS,
    REMOVED_FEATURES,
    apply_v2_2_ablation,
)

SCHEMA_VERSION = "1.0.0"
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
SPLITS = ("train", "validation", "test")
TRACKS = ("annotator_a", "annotator_b", "consensus")

SOURCE_HASHES = {
    "train": "18e1683e4afb43b07fd6a808c938739fd83a733dc275cc46049422ec9670f36a",
    "validation": "56624765518bf6aa65b24464c2bb325a4733fabe111c370dbb81830b5c6faf3c",
    "test": "406b2653555680178f9b07d6ca21c474a25cc0f88011c5780e55bebb9698c6b1",
}

PROVISIONAL_MAPPING = {
    "A1": "basic",
    "A2": "elementary_intermediate_english",
    "B1": "elementary_intermediate_english",
    "B2": "upper_intermediate_english",
    "C1": "advanced_english",
    "C2": "advanced_english",
}


@dataclass(frozen=True)
class _Record:
    sentence_id: str
    text: str
    text_hash: str
    annotator_a: str
    annotator_b: str
    consensus: str | None


@dataclass(frozen=True)
class _Example:
    sentence_id: str
    text: str
    label: str
    features: tuple[float, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Report already exists: {path}")

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
            raise FileExistsError(f"Report already exists: {path}")

        os.replace(temporary_path, path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_split(path: Path, split: str) -> tuple[list[_Record], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records = []
    seen_ids = set()
    seen_texts = {}
    duplicate_count = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"Unexpected blank JSONL line: {split}:{line_number}"
                )

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {split}:{line_number}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(f"Invalid record at {split}:{line_number}")

            sentence_id = item.get("sentence_id")
            text = item.get("text")
            text_hash = item.get("text_sha256_normalized")
            source = item.get("source")
            annotations = item.get("annotations")

            if not isinstance(sentence_id, str) or not sentence_id.strip():
                raise ValueError(f"Invalid sentence ID at {split}:{line_number}")

            if sentence_id in seen_ids:
                raise ValueError(f"Duplicate sentence ID: {sentence_id}")

            seen_ids.add(sentence_id)

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text at {split}:{line_number}")

            if not isinstance(source, dict) or not isinstance(annotations, dict):
                raise ValueError(f"Missing provenance or annotations at {sentence_id}")

            expected_file = {
                "train": "CEFR-SP_Wikiauto_train.txt",
                "validation": "CEFR-SP_Wikiauto_dev.txt",
                "test": "CEFR-SP_Wikiauto_test.txt",
            }[split]

            if (
                source.get("dataset") != "CEFR-SP Wiki-Auto"
                or source.get("license") != "CC BY-SA 3.0"
                or source.get("split") != split
                or source.get("source_file") != expected_file
                or source.get("source_file_sha256") != SOURCE_HASHES[split]
            ):
                raise ValueError(f"Invalid source provenance: {sentence_id}")

            if (
                not isinstance(source.get("source_line"), int)
                or isinstance(source.get("source_line"), bool)
                or source["source_line"] < 1
            ):
                raise ValueError(f"Invalid source line: {sentence_id}")

            calculated_hash = sentence_group_id(text)

            if text_hash != calculated_hash:
                raise ValueError(f"Text hash mismatch: {sentence_id}")

            labels = {}

            for annotator in ("annotator_a", "annotator_b"):
                annotation = annotations.get(annotator)

                if not isinstance(annotation, dict):
                    raise ValueError(
                        f"Missing {annotator} annotation: {sentence_id}"
                    )

                original_label = annotation.get("original_label")
                cefr_level = annotation.get("cefr_level")

                if (
                    isinstance(original_label, bool)
                    or not isinstance(original_label, int)
                    or not 1 <= original_label <= 6
                    or cefr_level != CEFR_LEVELS[original_label - 1]
                    or annotation.get("vyrex_provisional_level")
                    != PROVISIONAL_MAPPING[cefr_level]
                ):
                    raise ValueError(
                        f"Invalid {annotator} label: {sentence_id}"
                    )

                labels[annotator] = cefr_level

            exact = labels["annotator_a"] == labels["annotator_b"]

            if annotations.get("exact_agreement") is not exact:
                raise ValueError(f"Agreement mismatch: {sentence_id}")

            consensus = annotations.get("consensus_cefr_level")

            if consensus != (labels["annotator_a"] if exact else None):
                raise ValueError(f"Consensus mismatch: {sentence_id}")

            record = _Record(
                sentence_id=sentence_id,
                text=text,
                text_hash=calculated_hash,
                annotator_a=labels["annotator_a"],
                annotator_b=labels["annotator_b"],
                consensus=consensus,
            )

            previous = seen_texts.get(calculated_hash)

            if previous is not None:
                if (
                    previous.annotator_a != record.annotator_a
                    or previous.annotator_b != record.annotator_b
                ):
                    raise ValueError(
                        f"Duplicate sentence has conflicting labels: {sentence_id}"
                    )

                duplicate_count += 1
                continue

            seen_texts[calculated_hash] = record
            records.append(record)

    return records, {
        "source_records": len(seen_ids),
        "unique_records": len(records),
        "duplicate_records": duplicate_count,
        "source_file_sha256": SOURCE_HASHES[split],
        "normalized_text_sha256": _sha256_file(path),
    }


def _feature_vectors(text: str) -> dict[str, tuple[float, ...]]:
    raw = analyze_sentence_v2_2(text)
    vectors = {}

    for configuration in REMOVED_FEATURES:
        ablated = apply_v2_2_ablation(raw, configuration)
        vector = []

        for group in FEATURE_GROUPS:
            for name, value in ablated[group].items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"Invalid feature {group}.{name}"
                    )

                vector.append(float(value))

        if len(vector) != FEATURE_COUNTS[configuration]:
            raise ValueError(
                f"Feature count mismatch for {configuration}"
            )

        vectors[configuration] = tuple(vector)

    return vectors


def _make_examples(
    records: list[_Record],
    track: str,
    feature_cache: dict,
    configuration: str,
) -> list[_Example]:
    examples = []

    for record in records:
        label = getattr(record, track)

        if label is None:
            continue

        examples.append(
            _Example(
                sentence_id=record.sentence_id,
                text=record.text,
                label=label,
                features=feature_cache[record.text_hash][configuration],
            )
        )

    return examples


def _class_preflight(
    examples: list[_Example],
    n_splits: int,
    track: str,
) -> dict:
    counts = Counter(example.label for example in examples)
    groups = build_group_ids(examples)

    groups_per_class = {
        level: len({
            groups[index]
            for index, example in enumerate(examples)
            if example.label == level
        })
        for level in CEFR_LEVELS
    }

    missing = [level for level in CEFR_LEVELS if counts[level] == 0]

    insufficient = {
        level: count
        for level, count in groups_per_class.items()
        if count < n_splits
    }

    if missing or insufficient:
        raise ValueError(
            f"{track} training preflight failed; "
            f"missing classes={missing}; "
            f"insufficient groups={insufficient}"
        )

    return {
        "class_counts": dict(counts),
        "independent_groups_per_class": groups_per_class,
    }


def _evaluate_track(
    train_records: list[_Record],
    validation_records: list[_Record],
    test_records: list[_Record],
    track: str,
    feature_cache: dict,
    n_splits: int,
    random_state: int,
) -> dict:
    configurations = {}

    for configuration in REMOVED_FEATURES:
        train = _make_examples(
            train_records, track, feature_cache, configuration
        )
        validation = _make_examples(
            validation_records, track, feature_cache, configuration
        )

        preflight = _class_preflight(train, n_splits, track)

        cv = cross_validate_models(
            train,
            n_splits=n_splits,
            random_state=random_state,
        )

        models = create_models(random_state=random_state)
        model_results = {}

        if len({example.label for example in validation}) < 2:
            raise ValueError(
                f"{track}/validation has fewer than two classes"
            )

        validation_features = [
            list(example.features) for example in validation
        ]
        validation_labels = [
            example.label for example in validation
        ]

        for name, model in models.items():
            model.fit(
                [list(example.features) for example in train],
                [example.label for example in train],
            )

            predictions = model.predict(validation_features)

            model_results[name] = {
                "validation": evaluate_predictions(
                    validation_labels,
                    [str(value) for value in predictions],
                    labels=CEFR_LEVELS,
                )
            }

        configurations[configuration] = {
            "feature_count": FEATURE_COUNTS[configuration],
            "removed_features": sorted(
                REMOVED_FEATURES[configuration]
            ),
            "training_preflight": preflight,
            "cross_validation": {
                "metrics": cv["metrics"],
                "fold_reports": cv["fold_reports"],
                "evaluation_method": cv["evaluation_method"],
                "duplicate_text_leakage_checked": True,
            },
            "validation_evaluation": model_results,
            "test_evaluation": "reserved_until_model_selection_is_frozen",
        }

    return {
        "training_records": len(train),
        "validation_records": len(validation),
        "test_records": sum(
            getattr(record, track) is not None
            for record in test_records
        ),
        "configurations": configurations,
    }


def run_cefr_sp_v2_3_experiment(
    dataset_directory: str | Path,
    report_path: str | Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run independent annotator-aware CEFR-SP experiments."""

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

    source = Path(dataset_directory)
    destination = Path(report_path)

    if not source.is_dir():
        raise FileNotFoundError(source)

    if destination.exists():
        raise FileExistsError(f"Report already exists: {destination}")

    split_records = {}
    split_summaries = {}
    all_hashes = set()

    for split in SPLITS:
        records, summary = _read_split(
            source / f"{split}.jsonl",
            split,
        )

        hashes = {record.text_hash for record in records}

        overlap = all_hashes & hashes

        if overlap:
            raise ValueError(
                f"Cross-split duplicate leakage detected in {split}"
            )

        all_hashes.update(hashes)
        split_records[split] = records
        split_summaries[split] = summary

    feature_cache = {}

    for records in split_records.values():
        for record in records:
            if record.text_hash not in feature_cache:
                feature_cache[record.text_hash] = _feature_vectors(
                    record.text
                )

    results = {}

    for track in TRACKS:
        results[track] = _evaluate_track(
            split_records["train"],
            split_records["validation"],
            split_records["test"],
            track,
            feature_cache,
            n_splits,
            random_state,
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "CEFR-SP Wiki-Auto",
            "license": "CC BY-SA 3.0",
            "total_unique_records": sum(
                item["unique_records"]
                for item in split_summaries.values()
            ),
            "splits": split_summaries,
            "unique_feature_extractions": len(feature_cache),
            "feature_cache_reused_across_tracks": True,
            "labels": list(CEFR_LEVELS),
            "provisional_vyrex_mapping": PROVISIONAL_MAPPING,
        },
        "evaluation": {
            "tracks": list(TRACKS),
            "training_method": "stratified_group_cross_validation",
            "held_out_method": "train_on_full_training_split",
            "validation_used_for_training": False,
            "test_used_for_training": False,
            "duplicate_text_leakage_checked": True,
            "cross_split_leakage_checked": True,
            "n_splits": n_splits,
            "random_state": random_state,
        },
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        },
        "results": results,
        "interpretation": {
            "source_annotations_preserved": True,
            "labels_are_gold_standard": False,
            "confidence_calibrated": False,
            "production_model_approved": False,
            "commercial_use_approved": False,
            "note": (
                "CEFR-SP professional annotations are preserved as "
                "separate annotator perspectives. Consensus includes "
                "exact agreements only. Review CC BY-SA implications "
                "before commercial model training or distribution."
            ),
        },
    }

    _write_json_atomic(destination, report)

    return {
        "status": "completed",
        "report_path": str(destination),
        "unique_records": report["dataset"]["total_unique_records"],
        "feature_extractions": len(feature_cache),
        "tracks": list(TRACKS),
    }
