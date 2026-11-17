import pytest

from vyrex_ai.data.curriculum import (
    CURRICULUM_STAGES,
    CurriculumAssignment,
    cumulative_levels,
    load_curriculum_manifest,
    select_documents_for_stage,
    validate_assignment_coverage,
    validate_assignments,
    write_curriculum_manifest,
)
from vyrex_ai.data.readers import Document


def make_assignment(tmp_path, name, level):
    return CurriculumAssignment(
        source_path=str((tmp_path / name).resolve()),
        level=level,
        rationale=f"Reviewed classification for {name}",
    )


def test_curriculum_manifest_round_trip(tmp_path):
    assignments = [
        make_assignment(tmp_path, "basic.txt", "basic_english"),
        make_assignment(
            tmp_path,
            "advanced.txt",
            "advanced_english",
        ),
    ]

    manifest_path = tmp_path / "curriculum.json"

    written_path = write_curriculum_manifest(
        assignments,
        manifest_path,
    )

    loaded = load_curriculum_manifest(written_path)

    assert len(loaded) == 2

    for assignment in assignments:
        normalized = str(
            (tmp_path / assignment.source_path).resolve()
        ) if not __import__("pathlib").Path(
            assignment.source_path
        ).is_absolute() else assignment.source_path

        assert normalized in loaded


def test_duplicate_assignments_are_rejected(tmp_path):
    assignment = make_assignment(
        tmp_path,
        "duplicate.txt",
        "basic_english",
    )

    with pytest.raises(ValueError, match="Duplicate curriculum assignment"):
        validate_assignments([assignment, assignment])


def test_invalid_curriculum_level_is_rejected(tmp_path):
    assignment = make_assignment(
        tmp_path,
        "invalid.txt",
        "beginner_english",
    )

    with pytest.raises(ValueError, match="Unknown curriculum level"):
        validate_assignments([assignment])


def test_missing_assignment_coverage_is_rejected(tmp_path):
    assignment = make_assignment(
        tmp_path,
        "basic.txt",
        "basic_english",
    )

    assignments = validate_assignments([assignment])

    expected_paths = [
        assignment.source_path,
        str((tmp_path / "missing.txt").resolve()),
    ]

    with pytest.raises(ValueError, match="lack curriculum assignments"):
        validate_assignment_coverage(assignments, expected_paths)


def test_unknown_assignment_path_is_rejected(tmp_path):
    first = make_assignment(
        tmp_path,
        "first.txt",
        "basic_english",
    )
    second = make_assignment(
        tmp_path,
        "second.txt",
        "advanced_english",
    )

    assignments = validate_assignments([first, second])

    with pytest.raises(ValueError, match="do not match"):
        validate_assignment_coverage(
            assignments,
            [first.source_path],
        )


def test_cumulative_document_selection(tmp_path):
    levels = [
        "basic_english",
        "elementary_intermediate_english",
        "upper_intermediate_english",
        "advanced_english",
        "english_aptitude",
    ]

    documents = []
    assignments_list = []

    for index, level in enumerate(levels):
        source_path = str(
            (tmp_path / f"document_{index}.txt").resolve()
        )

        documents.append(
            Document(
                source_path=source_path,
                text=f"document_{index}",
            )
        )

        assignments_list.append(
            CurriculumAssignment(
                source_path=source_path,
                level=level,
                rationale=f"Reviewed level: {level}",
            )
        )

    assignments = validate_assignments(assignments_list)

    for index, stage in enumerate(CURRICULUM_STAGES):
        selected = select_documents_for_stage(
            documents,
            assignments,
            stage,
        )

        assert len(selected) == index + 1

    assert cumulative_levels("english_aptitude") == tuple(levels)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown curriculum stage"):
        cumulative_levels("unknown_stage")


def test_empty_rationale_is_rejected(tmp_path):
    assignment = CurriculumAssignment(
        source_path=str((tmp_path / "empty.txt").resolve()),
        level="basic_english",
        rationale="   ",
    )

    with pytest.raises(ValueError, match="rationale is required"):
        validate_assignments([assignment])


def test_load_curriculum_manifest_accepts_utf8_bom(tmp_path):
    import json

    from vyrex_ai.data.curriculum import load_curriculum_manifest

    document_path = (tmp_path / "sample.txt").resolve()
    manifest_path = tmp_path / "curriculum_bom.json"

    payload = {
        "schema_version": 1,
        "assignments": [
            {
                "source_path": str(document_path),
                "level": "basic_english",
                "rationale": "Regression test for BOM compatibility",
            }
        ],
    }

    manifest_path.write_bytes(
        bytes.fromhex("efbbbf")
        + json.dumps(payload).encode("utf-8")
    )

    assignments = load_curriculum_manifest(manifest_path)

    assert len(assignments) == 1
    assert assignments[str(document_path)] .level == "basic_english"
