"""Configuration loading and validation for VYREX AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_CURRICULUM_STAGES = (
    "basic_english",
    "elementary_intermediate_english",
    "upper_intermediate_english",
    "advanced_english",
    "english_aptitude",
)


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    max_seq_len: int
    positional_encoding: str
    norm: str
    mlp: str
    dropout: float
    tie_embeddings: bool


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    gradient_clip_norm: float
    precision: str
    log_every_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int
    num_workers: int


@dataclass(frozen=True)
class CurriculumConfig:
    stages: tuple[str, ...]
    progressive_training: bool
    evaluate_before_advancing: bool
    retain_previous_stage_knowledge: bool


@dataclass(frozen=True)
class DataConfig:
    tokenizer_name: str
    tokenizer_vocab_size: int
    train_split: float
    validation_split: float
    test_split: float
    mixture: dict[str, float]
    max_document_chars: int


@dataclass(frozen=True)
class ProjectConfig:
    project: dict[str, Any]
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: dict[str, str]
    curriculum: CurriculumConfig | None = None


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {section}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> ProjectConfig:
    """Load YAML configuration, construct typed sections, and validate invariants."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a YAML mapping")

    project = _require(raw, "project", "root")
    model_raw = _require(raw, "model", "root")
    train_raw = _require(raw, "training", "root")
    data_raw = _require(raw, "data", "root")
    paths = _require(raw, "paths", "root")

    if not all(isinstance(section, dict) for section in (
        project, model_raw, train_raw, data_raw, paths
    )):
        raise ValueError("Project, model, training, data, and paths must be mappings")

    model = ModelConfig(**model_raw)

    train_values = dict(train_raw)
    train_values["betas"] = tuple(train_values["betas"])
    training = TrainingConfig(**train_values)

    data = DataConfig(**data_raw)

    curriculum = None
    if "curriculum" in raw:
        curriculum_raw = raw["curriculum"]
        if not isinstance(curriculum_raw, dict):
            raise ValueError("curriculum must be a YAML mapping")

        curriculum_values = dict(curriculum_raw)
        curriculum_values["stages"] = tuple(curriculum_values["stages"])
        curriculum = CurriculumConfig(**curriculum_values)

        if curriculum.stages != EXPECTED_CURRICULUM_STAGES:
            raise ValueError(
                "Curriculum stages must contain the five approved stages "
                "in the approved order"
            )

        for field_name in (
            "progressive_training",
            "evaluate_before_advancing",
            "retain_previous_stage_knowledge",
        ):
            if not isinstance(getattr(curriculum, field_name), bool):
                raise TypeError(f"curriculum.{field_name} must be a boolean")

        if not all((
            curriculum.progressive_training,
            curriculum.evaluate_before_advancing,
            curriculum.retain_previous_stage_knowledge,
        )):
            raise ValueError("All approved progressive curriculum safeguards must be enabled")

    if model.hidden_size <= 0 or model.num_layers <= 0 or model.num_heads <= 0:
        raise ValueError("Model dimensions and layer/head counts must be positive")
    if model.hidden_size % model.num_heads != 0:
        raise ValueError("model.hidden_size must be divisible by model.num_heads")
    if (model.hidden_size // model.num_heads) % 2 != 0:
        raise ValueError("Attention head dimension must be even for RoPE")
    if model.vocab_size < 2 or model.max_seq_len < 2 or model.intermediate_size <= 0:
        raise ValueError("Model vocabulary, sequence length, and intermediate size are invalid")
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("model.dropout must be in the interval [0, 1)")
    if model.vocab_size != data.tokenizer_vocab_size:
        raise ValueError("model.vocab_size must match data.tokenizer_vocab_size")

    split_values = (
        data.train_split,
        data.validation_split,
        data.test_split,
    )
    if any(x < 0.0 or x > 1.0 for x in split_values):
        raise ValueError("Dataset split fractions must be between 0 and 1")
    if abs(sum(split_values) - 1.0) > 1e-6:
        raise ValueError("Dataset split fractions must sum to 1.0")

    if not data.mixture or any(v < 0.0 for v in data.mixture.values()):
        raise ValueError("data.mixture must contain non-negative proportions")
    if abs(sum(data.mixture.values()) - 1.0) > 1e-6:
        raise ValueError("data.mixture proportions must sum to 1.0")

    if curriculum is not None and set(data.mixture) != {"english", "aptitude"}:
        raise ValueError(
            "Curriculum-enabled configurations may contain only "
            "'english' and 'aptitude' data categories"
        )

    if training.learning_rate <= 0 or training.min_learning_rate < 0:
        raise ValueError("Learning rates must be positive/non-negative")
    if training.min_learning_rate > training.learning_rate:
        raise ValueError("training.min_learning_rate cannot exceed learning_rate")
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative")
    if training.micro_batch_size <= 0 or training.gradient_accumulation_steps <= 0:
        raise ValueError("Batch and gradient accumulation sizes must be positive")
    if training.max_steps <= 0 or not 0 <= training.warmup_steps < training.max_steps:
        raise ValueError("Training steps must be positive and warmup_steps < max_steps")
    if len(training.betas) != 2 or any(not 0.0 <= b < 1.0 for b in training.betas):
        raise ValueError("training.betas must contain two values in [0, 1)")
    if training.gradient_clip_norm < 0:
        raise ValueError("training.gradient_clip_norm cannot be negative")
    if training.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("training.precision must be fp32, fp16, or bf16")
    if any(value < 0 for value in (
        training.log_every_steps,
        training.eval_every_steps,
        training.checkpoint_every_steps,
        training.num_workers,
    )):
        raise ValueError("Training intervals and num_workers cannot be negative")

    if data.max_document_chars <= 0:
        raise ValueError("data.max_document_chars must be positive")

    return ProjectConfig(
        project=project,
        model=model,
        training=training,
        data=data,
        paths=paths,
        curriculum=curriculum,
    )
