from types import SimpleNamespace

import pytest

from vyrex_ai.data import english_level_v2_3_validation as validation


def make_example(sentence_id, text, label="Basic"):
    return SimpleNamespace(
        sentence_id=sentence_id,
        text=text,
        label=label,
        rationale="Label supported by linguistic evidence",
        reviewer="test-reviewer",
        reviewed_at="2026-11-13T10:00:00+05:30",
    )


def fake_cross_validate(examples, n_splits=5, random_state=42):
    return {
        "metrics": {"logistic_regression": {"macro_f1": 0.5}},
        "fold_reports": [],
        "evaluation_method": "stratified_group_cross_validation",
        "duplicate_text_leakage_checked": True,
    }


def test_v2_3_compares_all_feature_configurations(monkeypatch):
    monkeypatch.setattr(
        validation,
        "cross_validate_models",
        fake_cross_validate,
    )

    result = validation.run_v2_3_feature_validation([
        make_example("1", "Researchers carefully evaluated the results."),
        make_example("2", "Although difficult, the experiment succeeded."),
    ])

    assert result["status"] == "completed"
    assert result["schema_version"] == "2.3.0"
    assert result["sample_count"] == 2
    assert set(result["configurations"]) == {
        "full",
        "constant_removed",
        "redundancy_reduced",
    }


def test_v2_3_feature_vectors_have_expected_dimensions(monkeypatch):
    dimensions = []

    def capture(examples, **kwargs):
        dimensions.append(len(examples[0].features))
        return fake_cross_validate(examples, **kwargs)

    monkeypatch.setattr(validation, "cross_validate_models", capture)

    validation.run_v2_3_feature_validation([
        make_example("1", "The researchers evaluated the results."),
    ])

    assert dimensions == [21, 20, 17]


def test_v2_3_rejects_duplicate_sentence_ids(monkeypatch):
    monkeypatch.setattr(
        validation,
        "cross_validate_models",
        fake_cross_validate,
    )

    examples = [
        make_example("1", "First unique sentence."),
        make_example("1", "Second unique sentence."),
    ]

    with pytest.raises(ValueError, match="Duplicate sentence_id"):
        validation.run_v2_3_feature_validation(examples)


def test_v2_3_rejects_missing_review_provenance(monkeypatch):
    monkeypatch.setattr(
        validation,
        "cross_validate_models",
        fake_cross_validate,
    )

    example = make_example("1", "A valid example sentence.")
    example = SimpleNamespace(
        **{**vars(example), "reviewer": ""}
    )

    with pytest.raises(ValueError, match="reviewer"):
        validation.run_v2_3_feature_validation([example])


def test_v2_3_rejects_empty_input():
    with pytest.raises(ValueError, match="required"):
        validation.run_v2_3_feature_validation([])
