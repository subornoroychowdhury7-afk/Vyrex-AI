"""Export cumulative curriculum datasets while preserving existing splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .curriculum import (
    CURRICULUM_STAGES,
    CurriculumAssignment,
    cumulative_levels,
    select_documents_for_stage,
    validate_assignment_coverage,
    validate_assignments,
)


@dataclass(frozen=True)
class CurriculumExportResult:
    """Locations and document counts produced by a curriculum export."""

    output_directory: Path
    stage_directories: dict[str, Path]
    stage_counts: dict[str, dict[str, int]]


def _write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    """Write JSONL through a temporary file and atomic replacement."""
    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_curriculum_datasets(
    splits: Mapping[str, Iterable],
    assignments: Mapping[str, CurriculumAssignment],
    output_directory: str | Path,
) -> CurriculumExportResult:
    """Export all cumulative stages without changing original split membership.

    The input mapping must contain train, validation, and test document
    collections. Each accepted document must have exactly one curriculum
    assignment. Existing stage directories are never overwritten.
    """
    required_splits = ("train", "validation", "test")

    missing_splits = [
        split for split in required_splits if split not in splits
    ]
    if missing_splits:
        raise ValueError(
            "Missing required dataset split(s): "
            + ", ".join(missing_splits)
        )

    assignment_values = (
        assignments.values()
        if isinstance(assignments, Mapping)
        else assignments
    )
    normalized_assignments = validate_assignments(assignment_values)

    documents_by_split = {
        split: tuple(splits[split])
        for split in required_splits
    }

    all_documents = []
    seen_paths: set[str] = set()

    for split in required_splits:
        for document in documents_by_split[split]:
            source_path = str(Path(document.source_path).expanduser().resolve())

            if source_path in seen_paths:
                raise ValueError(
                    "A document appears in more than one split: "
                    f"{source_path}"
                )

            seen_paths.add(source_path)
            all_documents.append(document)

    validate_assignment_coverage(
        normalized_assignments,
        (document.source_path for document in all_documents),
    )

    output_root = Path(output_directory)
    stage_directories = {
        stage: output_root / stage
        for stage in CURRICULUM_STAGES
    }

    existing = [
        str(path)
        for path in stage_directories.values()
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing curriculum stage directory: "
            + existing[0]
        )

    output_root.mkdir(parents=True, exist_ok=True)

    stage_counts: dict[str, dict[str, int]] = {}

    for stage in CURRICULUM_STAGES:
        stage_directory = stage_directories[stage]
        stage_directory.mkdir(parents=True, exist_ok=False)

        counts: dict[str, int] = {}
        manifest_records = []

        for split in required_splits:
            selected = select_documents_for_stage(
                documents_by_split[split],
                normalized_assignments,
                stage,
            )

            # Preserve the existing text-only training record format.
            text_records = (
                {"text": document.text}
                for document in selected
            )

            _write_jsonl_atomic(
                stage_directory / f"{split}.jsonl",
                text_records,
            )
            counts[split] = len(selected)

            for document in selected:
                source_path = str(
                    Path(document.source_path).expanduser().resolve()
                )
                assignment = normalized_assignments[source_path]

                manifest_records.append(
                    {
                        "stage": stage,
                        "split": split,
                        "source_path": source_path,
                        "curriculum_level": assignment.level,
                        "rationale": assignment.rationale,
                    }
                )

        manifest_records.sort(
            key=lambda record: (
                record["split"],
                record["source_path"],
            )
        )

        _write_jsonl_atomic(
            stage_directory / "curriculum_manifest.jsonl",
            manifest_records,
        )

        stage_counts[stage] = counts

    return CurriculumExportResult(
        output_directory=output_root,
        stage_directories=stage_directories,
        stage_counts=stage_counts,
    )
