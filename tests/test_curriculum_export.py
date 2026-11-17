import json

import pytest

from vyrex_ai.data.curriculum import (
    CURRICULUM_STAGES,
    CurriculumAssignment,
)
from vyrex_ai.data.curriculum_export import (
    export_curriculum_datasets,
)
from vyrex_ai.data.readers import Document


def make_document(tmp_path, name, text):
    path = (tmp_path / name).resolve()
    return Document(source_path=str(path), text=text)


def make_assignments(splits, levels):
    assignments = []

    for split_documents in splits.values():
        for document in split_documents:
            name = document.text
            assignments.append(
                CurriculumAssignment(
                    source_path=document.source_path,
                    level=levels[name],
                    rationale=f"Test classification for {name}",
                )
            )

    return assignments


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_cumulative_export_preserves_splits(tmp_path):
    levels = {
        "basic": "basic_english",
        "elementary": "elementary_intermediate_english",
        "upper": "upper_intermediate_english",
        "advanced": "advanced_english",
        "aptitude": "english_aptitude",
    }

    splits = {
        "train": [
            make_document(tmp_path, f"{name}.txt", name)
            for name in levels
        ],
        "validation": [
            make_document(tmp_path, "val_basic.txt", "basic"),
            make_document(tmp_path, "val_advanced.txt", "advanced"),
        ],
        "test": [
            make_document(tmp_path, "test_elementary.txt", "elementary"),
            make_document(tmp_path, "test_aptitude.txt", "aptitude"),
        ],
    }

    # Keep source paths unique across all splits.
    splits["validation"][0] = make_document(
        tmp_path, "val_basic.txt", "basic"
    )

    assignments = make_assignments(splits, levels)

    output = tmp_path / "curriculum-output"

    result = export_curriculum_datasets(
        splits,
        assignments,
        output,
    )

    assert tuple(result.stage_directories) == CURRICULUM_STAGES

    expected_train_counts = [1, 2, 3, 4, 5]

    for index, stage in enumerate(CURRICULUM_STAGES):
        directory = result.stage_directories[stage]

        train_records = read_jsonl(directory / "train.jsonl")
        assert len(train_records) == expected_train_counts[index]

        assert all(
            list(record.keys()) == ["text"]
            for record in train_records
        )

        assert (directory / "validation.jsonl").is_file()
        assert (directory / "test.jsonl").is_file()
        assert (directory / "curriculum_manifest.jsonl").is_file()

    # Original split membership must remain unchanged.
    for stage in CURRICULUM_STAGES:
        manifest = read_jsonl(
            result.stage_directories[stage]
            / "curriculum_manifest.jsonl"
        )

        assert all(
            record["split"] in splits
            for record in manifest
        )


def test_export_rejects_missing_assignment(tmp_path):
    document = make_document(tmp_path, "missing.txt", "basic")

    splits = {
        "train": [document],
        "validation": [],
        "test": [],
    }

    with pytest.raises(ValueError, match="lack curriculum assignments"):
        export_curriculum_datasets(
            splits,
            [],
            tmp_path / "missing-output",
        )


def test_export_rejects_duplicate_split_membership(tmp_path):
    document = make_document(tmp_path, "duplicate.txt", "basic")

    splits = {
        "train": [document],
        "validation": [document],
        "test": [],
    }

    assignment = CurriculumAssignment(
        source_path=document.source_path,
        level="basic_english",
        rationale="Test assignment",
    )

    with pytest.raises(ValueError, match="more than one split"):
        export_curriculum_datasets(
            splits,
            [assignment],
            tmp_path / "duplicate-output",
        )


def test_export_refuses_existing_stage_directory(tmp_path):
    document = make_document(tmp_path, "existing.txt", "basic")

    splits = {
        "train": [document],
        "validation": [],
        "test": [],
    }

    assignment = CurriculumAssignment(
        source_path=document.source_path,
        level="basic_english",
        rationale="Test assignment",
    )

    output = tmp_path / "existing-output"
    existing_stage = output / CURRICULUM_STAGES[0]
    existing_stage.mkdir(parents=True)
    marker = existing_stage / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_curriculum_datasets(
            splits,
            [assignment],
            output,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"
