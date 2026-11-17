import math

import pytest

from vyrex_ai.data.english_level_cefr_sp_v2_3 import (
    CEFR_LEVELS,
    TRACKS,
    _Example,
    _class_preflight,
    _feature_vectors,
)


def test_six_cefr_levels_and_three_annotation_tracks():
    assert CEFR_LEVELS == ("A1", "A2", "B1", "B2", "C1", "C2")
    assert TRACKS == ("annotator_a", "annotator_b", "consensus")


def test_feature_extraction_returns_all_three_configurations():
    features = _feature_vectors(
        "Although the proposal was challenging, the team successfully completed it."
    )

    assert set(features) == {
        "full",
        "constant_removed",
        "redundancy_reduced",
    }
    assert len(features["full"]) == 21
    assert len(features["constant_removed"]) == 20
    assert len(features["redundancy_reduced"]) == 17

    for vector in features.values():
        assert all(math.isfinite(value) for value in vector)


def test_class_preflight_accepts_six_classes_with_enough_groups():
    examples = [
        _Example(
            sentence_id=f"{level}-{index}",
            text=f"Unique sentence for {level} group {index}.",
            label=level,
            features=(float(index), 1.0),
        )
        for level in CEFR_LEVELS
        for index in range(5)
    ]

    result = _class_preflight(examples, 5, "annotator_a")

    assert isinstance(result, dict)


def test_class_preflight_rejects_missing_cefr_classes():
    examples = [
        _Example(
            sentence_id=f"{level}-{index}",
            text=f"Unique sentence for {level} group {index}.",
            label=level,
            features=(float(index), 1.0),
        )
        for level in CEFR_LEVELS[:-1]
        for index in range(5)
    ]

    with pytest.raises(ValueError):
        _class_preflight(examples, 5, "annotator_a")


def test_class_preflight_rejects_insufficient_groups():
    examples = [
        _Example(
            sentence_id=f"{level}-{index}",
            text=f"Unique sentence for {level} group {index}.",
            label=level,
            features=(float(index), 1.0),
        )
        for level in CEFR_LEVELS
        for index in range(4)
    ]

    with pytest.raises(ValueError):
        _class_preflight(examples, 5, "annotator_a")
