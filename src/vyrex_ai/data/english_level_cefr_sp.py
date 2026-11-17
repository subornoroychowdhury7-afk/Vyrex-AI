"""Import and audit CEFR-SP Wiki-Auto annotations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
SOURCE_REPOSITORY = "https://github.com/yukiar/CEFR-SP"
SOURCE_LICENSE = "CC BY-SA 3.0"
SOURCE_CITATION = (
    "Yuki Arase, Satoru Uchida, and Tomoyuki Kajiwara. 2022. "
    "CEFR-Based Sentence-Difficulty Annotation and Assessment. EMNLP 2022."
)

SPLIT_FILES = {
    "train": "CEFR-SP_Wikiauto_train.txt",
    "validation": "CEFR-SP_Wikiauto_dev.txt",
    "test": "CEFR-SP_Wikiauto_test.txt",
}

CEFR_LEVELS = {
    1: "A1", 2: "A2", 3: "B1",
    4: "B2", 5: "C1", 6: "C2",
}

VYREX_LEVELS = {
    1: "basic",
    2: "elementary_intermediate_english",
    3: "elementary_intermediate_english",
    4: "upper_intermediate_english",
    5: "advanced_english",
    6: "advanced_english",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _weighted_kappa(
    matrix: list[list[int]],
    total: int,
    quadratic: bool,
) -> float | None:
    if not total:
        return None

    rows = [sum(row) for row in matrix]
    columns = [
        sum(matrix[i][j] for i in range(6))
        for j in range(6)
    ]

    def weight(i: int, j: int) -> float:
        distance = abs(i - j) / 5
        return 1 - (distance ** 2 if quadratic else distance)

    observed = sum(
        matrix[i][j] * weight(i, j)
        for i in range(6)
        for j in range(6)
    ) / total

    expected = sum(
        rows[i] * columns[j] / total * weight(i, j)
        for i in range(6)
        for j in range(6)
    ) / total

    if abs(1 - expected) < 1e-15:
        return None

    return (observed - expected) / (1 - expected)


def _agreement_report(
    matrix: list[list[int]],
    total: int,
) -> dict[str, Any]:
    exact = sum(matrix[i][i] for i in range(6))

    adjacent = sum(
        matrix[i][j]
        for i in range(6)
        for j in range(6)
        if abs(i - j) == 1
    )

    return {
        "annotated_records": total,
        "exact_agreement_count": exact,
        "exact_agreement_rate": exact / total if total else None,
        "adjacent_disagreement_count": adjacent,
        "adjacent_disagreement_rate": adjacent / total if total else None,
        "linear_weighted_cohen_kappa": _weighted_kappa(
            matrix, total, False
        ),
        "quadratic_weighted_cohen_kappa": _weighted_kappa(
            matrix, total, True
        ),
        "confusion_matrix": matrix,
        "matrix_axes": "Rows: annotator A; columns: annotator B; indices 1-6",
    }


def import_cefr_sp_wikiauto(
    source_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Import all Wiki-Auto splits without modifying source files."""

    source = Path(source_directory).resolve()
    destination = Path(output_directory).resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if destination.exists():
        raise FileExistsError(
            f"Output directory already exists: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    files = {
        split: source / filename
        for split, filename in SPLIT_FILES.items()
    }

    for split, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {split} file: {path}"
            )

    hashes = {
        split: _sha256_file(path)
        for split, path in files.items()
    }

    matrix = [[0] * 6 for _ in range(6)]
    seen_text: dict[str, set[str]] = {}
    split_stats = {}
    total_records = 0
    blank_lines = 0
    within_split_duplicates = 0
    cross_split_duplicate_records = 0

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary)

        for split, source_path in files.items():
            output_path = staging / f"{split}.jsonl"
            split_seen = set()
            stats = {"records": 0, "blank_lines": 0}

            with source_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as reader, output_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as writer:

                for line_number, raw_line in enumerate(reader, 1):
                    line = raw_line.rstrip("\r\n")

                    if not line.strip():
                        stats["blank_lines"] += 1
                        continue

                    fields = line.split("\t")

                    if len(fields) != 3:
                        raise ValueError(
                            f"Invalid TSV fields: {source_path}:{line_number}"
                        )

                    text, raw_a, raw_b = (
                        field.strip() for field in fields
                    )

                    if not text:
                        raise ValueError(
                            f"Empty sentence: {source_path}:{line_number}"
                        )

                    try:
                        label_a = int(raw_a)
                        label_b = int(raw_b)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid labels: {source_path}:{line_number}"
                        ) from exc

                    if label_a not in CEFR_LEVELS or label_b not in CEFR_LEVELS:
                        raise ValueError(
                            f"Labels must be 1..6: {source_path}:{line_number}"
                        )

                    digest = _text_hash(text)
                    prior_splits = seen_text.setdefault(digest, set())

                    if digest in split_seen:
                        within_split_duplicates += 1
                    elif prior_splits and split not in prior_splits:
                        cross_split_duplicate_records += 1

                    split_seen.add(digest)
                    prior_splits.add(split)

                    matrix[label_a - 1][label_b - 1] += 1
                    agreement = label_a == label_b

                    record = {
                        "sentence_id": (
                            f"cefr_sp_wikiauto:{split}:{line_number}"
                        ),
                        "text": text,
                        "text_sha256_normalized": digest,
                        "source": {
                            "dataset": "CEFR-SP Wiki-Auto",
                            "repository": SOURCE_REPOSITORY,
                            "license": SOURCE_LICENSE,
                            "split": split,
                            "source_file": source_path.name,
                            "source_line": line_number,
                            "source_file_sha256": hashes[split],
                        },
                        "annotations": {
                            "annotator_a": {
                                "original_label": label_a,
                                "cefr_level": CEFR_LEVELS[label_a],
                                "vyrex_provisional_level": VYREX_LEVELS[label_a],
                            },
                            "annotator_b": {
                                "original_label": label_b,
                                "cefr_level": CEFR_LEVELS[label_b],
                                "vyrex_provisional_level": VYREX_LEVELS[label_b],
                            },
                            "exact_agreement": agreement,
                            "consensus_cefr_level": (
                                CEFR_LEVELS[label_a] if agreement else None
                            ),
                            "consensus_vyrex_level": (
                                VYREX_LEVELS[label_a] if agreement else None
                            ),
                        },
                    }

                    writer.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ) + "\n"
                    )

                    stats["records"] += 1
                    total_records += 1

            split_stats[split] = stats
            blank_lines += stats["blank_lines"]

        cross_split_texts = sum(
            len(splits) > 1 for splits in seen_text.values()
        )

        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "dataset": "CEFR-SP Wiki-Auto",
                "repository": SOURCE_REPOSITORY,
                "license": SOURCE_LICENSE,
                "citation": SOURCE_CITATION,
                "source_file_sha256": hashes,
                "license_scope": (
                    "Wiki-Auto only. Other CEFR-SP subsets have "
                    "different licenses."
                ),
            },
            "mapping": {
                "cefr": {str(k): v for k, v in CEFR_LEVELS.items()},
                "vyrex_provisional": {
                    str(k): v for k, v in VYREX_LEVELS.items()
                },
                "mapping_is_official_cefr_equivalence": False,
                "disagreement_policy": (
                    "Preserve both annotations. Consensus is null "
                    "unless annotators agree exactly."
                ),
            },
            "dataset": {
                "total_records": total_records,
                "blank_lines": blank_lines,
                "splits": split_stats,
                "within_split_duplicate_records": within_split_duplicates,
                "cross_split_duplicate_records": cross_split_duplicate_records,
                "unique_texts_present_in_multiple_splits": cross_split_texts,
            },
            "annotation_agreement": _agreement_report(
                matrix, total_records
            ),
            "environment": {
                "python": platform.python_version()
            },
            "interpretation": {
                "human_annotated": True,
                "gold_standard": False,
                "training_approved": False,
                "production_approved": False,
                "note": (
                    "Preserve source annotations and investigate "
                    "disagreement. Review CC BY-SA implications before "
                    "commercial model training or distribution."
                ),
            },
        }

        (staging / "audit_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        (staging / "ATTRIBUTION.txt").write_text(
            f"Dataset: CEFR-SP Wiki-Auto\n"
            f"License: {SOURCE_LICENSE}\n"
            f"Source: {SOURCE_REPOSITORY}\n"
            f"Citation: {SOURCE_CITATION}\n",
            encoding="utf-8",
        )

        if destination.exists():
            raise FileExistsError(
                f"Output directory already exists: {destination}"
            )

        os.replace(staging, destination)

    report["output_directory"] = str(destination)
    return report
