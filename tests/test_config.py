from pathlib import Path

import pytest
import yaml

from vyrex_ai.config import EXPECTED_CURRICULUM_STAGES, load_config

ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG = ROOT / "configs" / "base-v2.yaml"


def test_base_config_loads_and_has_expected_shape():
    config = load_config(ROOT / "configs" / "base.yaml")
    assert config.project["name"] == "VYREX AI"
    assert config.project["mode"] == "hybrid"
    assert config.project["language"] == "en"
    assert config.model.hidden_size % config.model.num_heads == 0
    assert config.model.vocab_size == config.data.tokenizer_vocab_size
    assert sum(config.data.mixture.values()) == pytest.approx(1.0)


def test_smoke_config_loads():
    config = load_config(ROOT / "configs" / "smoke.yaml")
    assert config.model.num_layers == 2
    assert config.model.max_seq_len == 128


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "configs" / "does-not-exist.yaml")


def test_v1_config_remains_compatible():
    config = load_config(ROOT / "configs" / "base-v1.yaml")
    assert config.curriculum is None
    assert "coding" in config.data.mixture


def test_v2_config_has_approved_curriculum():
    config = load_config(V2_CONFIG)
    assert config.curriculum is not None
    assert config.curriculum.stages == EXPECTED_CURRICULUM_STAGES
    assert config.curriculum.progressive_training is True
    assert config.curriculum.evaluate_before_advancing is True
    assert config.curriculum.retain_previous_stage_knowledge is True


def test_v2_config_has_english_and_aptitude_only():
    config = load_config(V2_CONFIG)
    assert set(config.data.mixture) == {"english", "aptitude"}
    assert config.data.mixture["english"] == pytest.approx(0.90)
    assert config.data.mixture["aptitude"] == pytest.approx(0.10)


def test_reordered_curriculum_stages_are_rejected(tmp_path):
    raw = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8-sig"))
    raw["curriculum"]["stages"][0], raw["curriculum"]["stages"][1] = (
        raw["curriculum"]["stages"][1],
        raw["curriculum"]["stages"][0],
    )
    invalid_path = tmp_path / "reordered.yaml"
    invalid_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="approved stages"):
        load_config(invalid_path)


def test_coding_mixture_is_rejected_with_curriculum(tmp_path):
    raw = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8-sig"))
    raw["data"]["mixture"]["coding"] = 0.10
    raw["data"]["mixture"]["english"] = 0.80
    invalid_path = tmp_path / "coding.yaml"
    invalid_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="only 'english' and 'aptitude'"):
        load_config(invalid_path)
