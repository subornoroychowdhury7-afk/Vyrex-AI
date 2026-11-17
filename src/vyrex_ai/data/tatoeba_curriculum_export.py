"""Leakage-safe export of human-reviewed Tatoeba curriculum datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .curriculum import CURRICULUM_STAGES, cumulative_levels
from .readers import Document
from .splitting import split_documents
from .tatoeba_curriculum import (
    TATOEBA_LEVELS,
    load_tatoeba_assignments,
)


ENGLISH_STAGES = CURRICULUM_STAGES[:4]
NORMALIZE_WHITESPACE = re.compile(r"\s+")


def _group_id(text: str) -> str:
    """Stable group ID for exact normalized-text duplicate protection."""
    normalized = NORMALIZE_WHITESPACE.sub(" ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def export_reviewed_tatoeba(
    accepted_path: str | Path,
    assignments_path: str | Path,
    output_directory: str | Path,
    *,
    train_ratio: float = 0.98,
    validation_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> dict:
    """Validate reviewed assignments, split by duplicate-text group, export.

    Only human-assigned accepted records are included. A single deterministic
    split is computed and reused for every cumulative English stage.
    The fifth aptitude stage is intentionally not exported from Tatoeba.
    """
    source = Path(accepted_path)
    assignment_file = Path(assignments_path)
    destination = Path(output_directory)

    if not source.is_file():
        raise FileNotFoundError(source)

    assignments = load_tatoeba_assignments(assignment_file)
    if not assignments:
        raise ValueError("No human-reviewed assignments were supplied.")

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    documents: list[Document] = []
    source_groups: dict[str, str] = {}
    records_by_id: dict[str, dict] = {}
    found_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed accepted record at line {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Accepted record at line {line_number} is not an object."
                )

            sentence_id = record.get("sentence_id")
            if not isinstance(sentence_id, str) or not sentence_id.strip():
                raise ValueError(
                    f"Missing sentence_id at line {line_number}."
                )

            if sentence_id not in assignments:
                continue

            if sentence_id in found_ids:
                raise ValueError(
                    f"Duplicate assigned sentence ID in source: {sentence_id}"
                )
            found_ids.add(sentence_id)

            assignment = assignments[sentence_id]

            if record.get("quality_bucket") != "accepted":
                raise ValueError(
                    f"Sentence {sentence_id} is not quality-approved."
                )

            if record.get("source_sha256", "").lower() != assignment.source_sha256:
                raise ValueError(
                    f"Source hash mismatch for sentence {sentence_id}."
                )

            if record.get("license") != assignment.license:
                raise ValueError(
                    f"License mismatch for sentence {sentence_id}."
                )

            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Empty text for assigned sentence {sentence_id}."
                )

            if assignment.level not in TATOEBA_LEVELS:
                raise ValueError(
                    f"Unsupported English level for {sentence_id}."
                )

            license_name = record["license"]
            if license_name == "CC-BY-2.0-FR":
                attribution = record.get("attribution")
                author = record.get("author")

                if (
                    not isinstance(author, str)
                    or not author.strip()
                    or not isinstance(attribution, dict)
                    or attribution.get("author") != author
                    or attribution.get("source") != "Tatoeba"
                ):
                    raise ValueError(
                        f"Missing or inconsistent CC BY attribution: {sentence_id}"
                    )
            elif license_name != "CC0-1.0":
                raise ValueError(
                    f"Unsupported license for sentence {sentence_id}: "
                    f"{license_name}"
                )

            internal_path = f"tatoeba://{sentence_id}"
            document = Document(
                source_path=internal_path,
                text=text,
            )

            documents.append(document)
            source_groups[internal_path] = _group_id(text)
            records_by_id[sentence_id] = record

    missing_ids = sorted(set(assignments) - found_ids)
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} assigned sentence(s) were not found. "
            f"First missing ID: {missing_ids[0]}"
        )

    if not documents:
        raise ValueError("No reviewed Tatoeba records were found.")

    splits = split_documents(
        documents,
        source_groups=source_groups,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_documents_by_name = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    split_by_id: dict[str, str] = {}
    group_splits: dict[str, set[str]] = {}

    for split_name, split_docs in split_documents_by_name.items():
        for document in split_docs:
            sentence_id = document.source_path.removeprefix("tatoeba://")
            split_by_id[sentence_id] = split_name
            group_splits.setdefault(
                source_groups[document.source_path], set()
            ).add(split_name)

    leaked_groups = [
        group_id for group_id, names in group_splits.items()
        if len(names) > 1
    ]
    if leaked_groups:
        raise RuntimeError(
            f"Detected {len(leaked_groups)} duplicate-text group leakage(s)."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    ))

    stage_counts: dict[str, dict[str, int]] = {}

    try:
        for stage in ENGLISH_STAGES:
            included_levels = set(cumulative_levels(stage))
            stage_directory = staging / stage
            stage_directory.mkdir()

            counts = {}
            manifest_records = []

            for split_name in ("train", "validation", "test"):
                selected_ids = [
                    sentence_id
                    for sentence_id, split_name_for_id in split_by_id.items()
                    if split_name_for_id == split_name
                    and assignments[sentence_id].level in included_levels
                ]

                selected_ids.sort()

                text_records = (
                    {"text": records_by_id[sentence_id]["text"]}
                    for sentence_id in selected_ids
                )
                _write_jsonl(
                    stage_directory / f"{split_name}.jsonl",
                    text_records,
                )
                counts[split_name] = len(selected_ids)

                for sentence_id in selected_ids:
                    record = records_by_id[sentence_id]
                    assignment = assignments[sentence_id]

                    manifest_records.append({
                        "sentence_id": sentence_id,
                        "split": split_name,
                        "curriculum_level": assignment.level,
                        "rationale": assignment.rationale,
                        "reviewer": assignment.reviewer,
                        "reviewed_at": assignment.reviewed_at,
                        "license": record["license"],
                        "author": record.get("author"),
                        "attribution": record.get("attribution"),
                        "source_archive": record.get("source_archive"),
                        "source_sha256": record["source_sha256"],
                        "text_sha256": hashlib.sha256(
                            record["text"].encode("utf-8")
                        ).hexdigest(),
                        "duplicate_group": source_groups[
                            f"tatoeba://{sentence_id}"
                        ],
                    })

            manifest_records.sort(
                key=lambda item: (item["split"], item["sentence_id"])
            )
            _write_jsonl(
                stage_directory / "curriculum_manifest.jsonl",
                manifest_records,
            )
            stage_counts[stage] = counts

        summary = {
            "schema_version": 1,
            "source": str(source.resolve()),
            "assignment_manifest": str(assignment_file.resolve()),
            "reviewed_sentence_count": len(documents),
            "split_counts": {
                name: len(items)
                for name, items in split_documents_by_name.items()
            },
            "stage_counts": stage_counts,
            "split_ratios": {
                "train": train_ratio,
                "validation": validation_ratio,
                "test": test_ratio,
            },
            "split_seed": seed,
            "duplicate_grouping": "SHA256 of casefolded whitespace-normalized text",
            "cross_split_duplicate_group_leaks": len(leaked_groups),
            "licenses": dict(sorted(Counter(
                record["license"] for record in records_by_id.values()
            ).items())),
            "aptitude_stage": {
                "status": "blocked_pending_original_aptitude_dataset",
                "note": "Tatoeba English sentences are not aptitude questions.",
            },
        }

        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        os.replace(staging, destination)
        return summary

    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

