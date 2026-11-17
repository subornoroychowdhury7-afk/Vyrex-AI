import json

import pytest

from vyrex_ai.data.english_level_cefr_sp import (
    CEFR_LEVELS,
    import_cefr_sp_wikiauto,
)


def create_source(root):
    root.mkdir(parents=True)

    (root / "CEFR-SP_Wikiauto_train.txt").write_text(
        "A simple sentence.\t1\t1\n"
        "A difficult sentence.\t5\t6\n",
        encoding="utf-8",
    )
    (root / "CEFR-SP_Wikiauto_dev.txt").write_text(
        "A simple sentence.\t1\t2\n",
        encoding="utf-8",
    )
    (root / "CEFR-SP_Wikiauto_test.txt").write_text(
        "Another sentence for testing.\t3\t3\n",
        encoding="utf-8",
    )


def test_import_preserves_both_annotators_and_mappings(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    create_source(source)

    result = import_cefr_sp_wikiauto(source, output)

    records = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 2
    assert records[0]["annotations"]["annotator_a"]["cefr_level"] == "A1"
    assert records[0]["annotations"]["annotator_b"]["cefr_level"] == "A1"
    assert records[0]["annotations"]["consensus_cefr_level"] == "A1"
    assert records[1]["annotations"]["annotator_a"]["cefr_level"] == "C1"
    assert records[1]["annotations"]["annotator_b"]["cefr_level"] == "C2"
    assert records[1]["annotations"]["consensus_cefr_level"] is None
    assert result["dataset"]["total_records"] == 4
    assert (output / "audit_report.json").is_file()
    assert (output / "ATTRIBUTION.txt").is_file()


def test_agreement_metrics_and_confusion_matrix(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    create_source(source)

    result = import_cefr_sp_wikiauto(source, output)
    agreement = result["annotation_agreement"]

    assert agreement["annotated_records"] == 4
    assert agreement["exact_agreement_count"] == 2
    assert agreement["adjacent_disagreement_count"] == 2
    assert agreement["quadratic_weighted_cohen_kappa"] is not None
    assert len(agreement["confusion_matrix"]) == 6
    assert all(len(row) == 6 for row in agreement["confusion_matrix"])


def test_duplicate_detection_across_splits(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    create_source(source)

    result = import_cefr_sp_wikiauto(source, output)
    stats = result["dataset"]

    assert stats["cross_split_duplicate_records"] == 1
    assert stats["unique_texts_present_in_multiple_splits"] == 1
    assert stats["within_split_duplicate_records"] == 0


def test_source_hashes_are_recorded(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    create_source(source)

    result = import_cefr_sp_wikiauto(source, output)

    assert set(result["source"]["source_file_sha256"]) == {
        "train", "validation", "test"
    }
    assert all(
        len(value) == 64
        for value in result["source"]["source_file_sha256"].values()
    )


def test_malformed_tsv_fails_without_publishing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    (source / "CEFR-SP_Wikiauto_train.txt").write_text(
        "Invalid row\t1\n", encoding="utf-8"
    )
    (source / "CEFR-SP_Wikiauto_dev.txt").write_text(
        "Valid sentence.\t1\t1\n", encoding="utf-8"
    )
    (source / "CEFR-SP_Wikiauto_test.txt").write_text(
        "Valid sentence.\t2\t2\n", encoding="utf-8"
    )

    output = tmp_path / "output"

    with pytest.raises(ValueError, match="Invalid TSV fields"):
        import_cefr_sp_wikiauto(source, output)

    assert not output.exists()


def test_invalid_label_fails_without_publishing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    (source / "CEFR-SP_Wikiauto_train.txt").write_text(
        "Invalid label.\t1\t7\n", encoding="utf-8"
    )
    (source / "CEFR-SP_Wikiauto_dev.txt").write_text(
        "Valid sentence.\t1\t1\n", encoding="utf-8"
    )
    (source / "CEFR-SP_Wikiauto_test.txt").write_text(
        "Valid sentence.\t2\t2\n", encoding="utf-8"
    )

    output = tmp_path / "output"

    with pytest.raises(ValueError, match="Labels must be 1..6"):
        import_cefr_sp_wikiauto(source, output)

    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    create_source(source)
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        import_cefr_sp_wikiauto(source, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_empty_splits_are_supported(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    for filename in (
        "CEFR-SP_Wikiauto_train.txt",
        "CEFR-SP_Wikiauto_dev.txt",
        "CEFR-SP_Wikiauto_test.txt",
    ):
        (source / filename).write_text("", encoding="utf-8")

    output = tmp_path / "output"
    result = import_cefr_sp_wikiauto(source, output)

    assert result["dataset"]["total_records"] == 0
    assert result["annotation_agreement"]["exact_agreement_rate"] is None
