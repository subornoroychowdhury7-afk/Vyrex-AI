"""Explicit curriculum assignments and cumulative stage selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CURRICULUM_STAGES = (
    "basic_english",
    "elementary_intermediate_english",
    "upper_intermediate_english",
    "advanced_english",
    "english_aptitude",
)

CURRICULUM_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CurriculumAssignment:
    """Reviewed curriculum classification for one source document."""

    source_path: str
    level: str
    rationale: str


def _normalize_source_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Curriculum source_path must be a non-empty string.")

    path = Path(value).expanduser()

    if not path.is_absolute():
        raise ValueError(
            f"Curriculum source_path must be absolute: {value}"
        )

    return str(path.resolve())


def validate_assignments(
    assignments: Iterable[CurriculumAssignment],
) -> dict[str, CurriculumAssignment]:
    """Validate assignments and return them indexed by normalized path."""
    indexed: dict[str, CurriculumAssignment] = {}

    for assignment in assignments:
        if not isinstance(assignment, CurriculumAssignment):
            raise TypeError(
                "Every curriculum assignment must be a "
                "CurriculumAssignment."
            )

        source_path = _normalize_source_path(assignment.source_path)

        if assignment.level not in CURRICULUM_STAGES:
            raise ValueError(
                f"Unknown curriculum level '{assignment.level}' "
                f"for {source_path}."
            )

        if not isinstance(assignment.rationale, str) or not assignment.rationale.strip():
            raise ValueError(
                f"Curriculum rationale is required for {source_path}."
            )

        if source_path in indexed:
            raise ValueError(
                f"Duplicate curriculum assignment for {source_path}."
            )

        indexed[source_path] = CurriculumAssignment(
            source_path=source_path,
            level=assignment.level,
            rationale=assignment.rationale.strip(),
        )

    return indexed


def load_curriculum_manifest(
    manifest_path: str | Path,
) -> dict[str, CurriculumAssignment]:
    """Load and validate a versioned JSON curriculum manifest."""
    path = Path(manifest_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Curriculum manifest not found: {path}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid curriculum manifest JSON: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Curriculum manifest must contain a JSON object.")

    if payload.get("schema_version") != CURRICULUM_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported curriculum manifest schema version."
        )

    raw_assignments = payload.get("assignments")

    if not isinstance(raw_assignments, list):
        raise ValueError(
            "Curriculum manifest 'assignments' must be a JSON array."
        )

    assignments: list[CurriculumAssignment] = []

    for index, item in enumerate(raw_assignments):
        if not isinstance(item, dict):
            raise ValueError(
                f"Assignment at index {index} must be a JSON object."
            )

        try:
            assignments.append(
                CurriculumAssignment(
                    source_path=item["source_path"],
                    level=item["level"],
                    rationale=item["rationale"],
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Assignment at index {index} is missing {exc.args[0]}."
            ) from exc

    return validate_assignments(assignments)


def write_curriculum_manifest(
    assignments: Iterable[CurriculumAssignment],
    output_path: str | Path,
) -> Path:
    """Write validated curriculum assignments using atomic replacement."""
    indexed = validate_assignments(assignments)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    payload = {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "assignments": [
            asdict(indexed[source_path])
            for source_path in sorted(indexed)
        ],
    }

    try:
        with temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return destination


def cumulative_levels(stage: str) -> tuple[str, ...]:
    """Return all curriculum levels included through the requested stage."""
    if stage not in CURRICULUM_STAGES:
        raise ValueError(f"Unknown curriculum stage: {stage}")

    stage_index = CURRICULUM_STAGES.index(stage)
    return CURRICULUM_STAGES[:stage_index + 1]


def validate_assignment_coverage(
    assignments: dict[str, CurriculumAssignment],
    source_paths: Iterable[str],
) -> None:
    """Require exactly one assignment for every supplied training document."""
    expected = {
        _normalize_source_path(source_path)
        for source_path in source_paths
    }
    actual = set(assignments)

    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)

    if missing:
        raise ValueError(
            f"{len(missing)} training document(s) lack curriculum assignments. "
            f"First missing path: {missing[0]}"
        )

    if unknown:
        raise ValueError(
            f"{len(unknown)} curriculum assignment(s) do not match "
            f"the supplied dataset. First unknown path: {unknown[0]}"
        )


def select_documents_for_stage(
    documents: Iterable,
    assignments: dict[str, CurriculumAssignment],
    stage: str,
):
    """Select documents belonging to the cumulative requested stage."""
    levels = set(cumulative_levels(stage))
    selected = []

    for document in documents:
        source_path = _normalize_source_path(document.source_path)
        assignment = assignments.get(source_path)

        if assignment is None:
            raise ValueError(
                f"Missing curriculum assignment for {source_path}."
            )

        if assignment.level in levels:
            selected.append(document)

    return tuple(selected)
