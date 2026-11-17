"""Command-line interface for VYREX AI project tools."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from vyrex_ai import __version__
from vyrex_ai.config import load_config
from vyrex_ai.data import export_dataset, prepare_dataset
from vyrex_ai.data.corpus import inspect_training_corpus, iter_training_texts
from vyrex_ai.logging_utils import configure_logging
from vyrex_ai.runtime import inspect_runtime, print_runtime_report
from vyrex_ai.tokenizer import DEFAULT_VOCAB_SIZE, VyrexTokenizer
from vyrex_ai.tokenizer_artifacts import save_tokenizer_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vyrex",
        description="VYREX AI project tools",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"VYREX AI {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect Python, PyTorch, and GPU availability",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    show = subparsers.add_parser(
        "show-config",
        help="Validate and summarize a YAML config",
    )
    show.add_argument("--config", default="configs/base.yaml")

    init = subparsers.add_parser(
        "init",
        help="Create configured project output directories",
    )
    init.add_argument("--config", default="configs/base.yaml")

    prepare = subparsers.add_parser(
        "prepare-data",
        help="Validate, split, and export a dataset",
    )
    prepare.add_argument("--input", required=True, help="Raw data directory")
    prepare.add_argument("--output", required=True, help="Processed output directory")
    prepare.add_argument(
        "--groups",
        required=True,
        help="JSON mapping of relative file paths to source groups",
    )
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-ratio", type=float, default=0.98)
    prepare.add_argument("--validation-ratio", type=float, default=0.01)
    prepare.add_argument("--test-ratio", type=float, default=0.01)
    prepare.add_argument(
        "--exclude-list",
        help="Optional JSON report listing documents to quarantine",
    )

    train_tokenizer = subparsers.add_parser(
        "train-tokenizer",
        help="Train and save a tokenizer using only train.jsonl",
    )
    train_tokenizer.add_argument(
        "--dataset",
        required=True,
        help="Prepared dataset directory containing train.jsonl",
    )
    train_tokenizer.add_argument(
        "--output",
        required=True,
        help="Directory for tokenizer artifacts",
    )
    train_tokenizer.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset identifier recorded in tokenizer metadata",
    )
    train_tokenizer.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )
    train_tokenizer.add_argument(
        "--min-frequency",
        type=int,
        default=2,
    )
    train_tokenizer.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing existing tokenizer artifacts",
    )

    train = subparsers.add_parser(
        "train",
        help="Train the decoder-only language model",
    )
    train.add_argument(
        "--config",
        default="configs/smoke.yaml",
        help="Training configuration YAML (defaults to the safe smoke config)",
    )
    train.add_argument(
        "--dataset",
        required=True,
        help="Prepared dataset directory containing train.jsonl and validation.jsonl",
    )
    train.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer.json",
    )
    train.add_argument(
        "--checkpoint-dir",
        required=True,
        help="New or explicitly resumed checkpoint directory",
    )
    train.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Training device (defaults to CUDA when available, otherwise CPU)",
    )
    train.add_argument(
        "--resume",
        help="Resume from a checkpoint located inside --checkpoint-dir",
    )
    train.add_argument(
        "--max-validation-batches",
        type=int,
        help="Optional limit on validation batches per evaluation",
    )


    train_curriculum_parser = subparsers.add_parser(
        "train-curriculum",
        help="Train VYREX AI through all five curriculum stages",
    )
    train_curriculum_parser.add_argument(
        "--config",
        default="configs/base-v2.yaml",
        help="Curriculum-enabled YAML configuration",
    )
    train_curriculum_parser.add_argument(
        "--dataset",
        required=True,
        help="Directory containing all five exported curriculum stages",
    )
    train_curriculum_parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the trained tokenizer JSON",
    )
    train_curriculum_parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory for curriculum checkpoints",
    )
    train_curriculum_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )
    train_curriculum_parser.add_argument(
        "--resume",
        default=None,
        help="Resume from latest.pt in the checkpoint directory",
    )
    train_curriculum_parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
    )
    train_curriculum_parser.add_argument(
        "--num-workers",
        type=int,
        choices=(0,),
        required=True,
        help="Must be 0 until multi-worker streaming is implemented",
    )

    evaluate = subparsers.add_parser(
        "evaluate-english-levels",
        help="Compare English-level ML classifiers using human-approved labels",
    )
    evaluate.add_argument(
        "--dataset",
        required=True,
        help="Human-reviewed JSONL dataset",
    )
    evaluate.add_argument(
        "--report",
        required=True,
        help="New JSON experiment report path",
    )
    evaluate.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Stratified group cross-validation folds (default: 5)",
    )
    evaluate.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducible random seed (default: 42)",
    )

    v23 = subparsers.add_parser(
        "evaluate-english-level-v2-3",
        help="Validate V2.2 feature configurations using human-approved labels",
    )
    v23.add_argument(
        "--dataset",
        required=True,
        help="Human-reviewed JSONL dataset",
    )
    v23.add_argument(
        "--report",
        required=True,
        help="New JSON V2.3 experiment report path",
    )
    v23.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Stratified group cross-validation folds (default: 5)",
    )
    v23.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducible random seed (default: 42)",
    )

    diagnostics = subparsers.add_parser(
        "evaluate-english-level-diagnostics",
        help="Compare V2 and V2.1 English diagnostic features",
    )
    diagnostics.add_argument(
        "--dataset",
        required=True,
        help="Input sentence JSONL dataset",
    )
    diagnostics.add_argument(
        "--report",
        required=True,
        help="New JSON diagnostic summary path",
    )
    diagnostics.add_argument(
        "--records",
        required=True,
        help="New JSONL record-level diagnostic output",
    )
    diagnostics.add_argument(
        "--maximum-records",
        type=int,
        default=1000,
        help="Maximum nonblank records to inspect (default: 1000)",
    )
    v22_audit = subparsers.add_parser(
        "audit-english-level-v2-2",
        help="Audit V2.2 feature quality, correlations and throughput",
    )
    v22_audit.add_argument(
        "--dataset",
        required=True,
        help="Input sentence JSONL dataset",
    )
    v22_audit.add_argument(
        "--report",
        required=True,
        help="New JSON audit report path",
    )
    v22_audit.add_argument(
        "--reference-dataset",
        help="Optional JSONL dataset for sentence ID and text overlap",
    )
    v22_audit.add_argument(
        "--maximum-records",
        type=int,
        default=5000,
        help="Maximum records sampled (default: 5000)",
    )
    v22_audit.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42)",
    )
    v22_audit.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="Absolute Pearson correlation threshold (default: 0.90)",
    )

    v22_ablation = subparsers.add_parser(
        "benchmark-english-level-v2-2-ablation",
        help="Benchmark V2.2 feature ablation configurations",
    )
    v22_ablation.add_argument(
        "--dataset",
        required=True,
        help="Input sentence JSONL dataset",
    )
    v22_ablation.add_argument(
        "--report",
        required=True,
        help="New JSON benchmark report path",
    )
    v22_ablation.add_argument(
        "--maximum-records",
        type=int,
        default=5000,
        help="Maximum records sampled (default: 5000)",
    )
    v22_ablation.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42)",
    )
    v22_ablation.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="Absolute Pearson correlation threshold (default: 0.90)",
    )

    expand = subparsers.add_parser(
        "expand-english-levels",
        help="Expand accepted Tatoeba sentences into a provisional review queue",
    )
    expand.add_argument(
        "--input",
        required=True,
        help="Accepted Tatoeba V3 JSONL input",
    )
    expand.add_argument(
        "--output",
        help="New candidate JSONL output (required outside dry-run)",
    )
    expand.add_argument(
        "--manifest",
        help="New expansion manifest JSONL output (required outside dry-run)",
    )
    expand.add_argument(
        "--maximum-records",
        type=int,
        default=10_000,
        help="Maximum nonblank records to examine (default: 10000)",
    )
    expand.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and classify without publishing output files",
    )

    cefr_sp = subparsers.add_parser(
        "import-cefr-sp-wikiauto",
        help="Import and audit CEFR-SP Wiki-Auto annotations",
    )
    cefr_sp.add_argument(
        "--source",
        required=True,
        help="Directory containing the original Wiki-Auto train/dev/test TSV files",
    )
    cefr_sp.add_argument(
        "--output",
        required=True,
        help="New output directory for imported JSONL splits and audit report",
    )

    cefr_sp_v23 = subparsers.add_parser(
        "evaluate-english-level-cefr-sp-v2-3",
        help="Evaluate CEFR-SP using separate annotator and consensus tracks",
    )
    cefr_sp_v23.add_argument(
        "--dataset",
        required=True,
        help="Imported CEFR-SP Wiki-Auto directory",
    )
    cefr_sp_v23.add_argument(
        "--report",
        required=True,
        help="New JSON evaluation report path",
    )
    cefr_sp_v23.add_argument("--folds", type=int, default=5)
    cefr_sp_v23.add_argument("--seed", type=int, default=42)

    return parser


def load_source_groups(
    groups_file: str | Path,
    input_directory: str | Path,
) -> dict[str, str]:
    """Load relative source paths and convert them to resolved paths."""
    groups_path = Path(groups_file)
    input_root = Path(input_directory).resolve()

    raw_mapping = json.loads(groups_path.read_text(encoding="utf-8"))

    if not isinstance(raw_mapping, dict):
        raise TypeError("Source-group mapping must be a JSON object.")

    resolved_mapping: dict[str, str] = {}

    for relative_path, group in raw_mapping.items():
        if not isinstance(relative_path, str) or not isinstance(group, str):
            raise TypeError("Every mapping key and group name must be a string.")

        if not relative_path.strip() or not group.strip():
            raise ValueError("Source paths and group names cannot be empty.")

        candidate = Path(relative_path)

        if candidate.is_absolute():
            raise ValueError(
                f"Source mapping paths must be relative: {relative_path}"
            )

        resolved_path = (input_root / candidate).resolve()

        if not resolved_path.is_relative_to(input_root):
            raise ValueError(
                f"Source path escapes the input directory: {relative_path}"
            )

        resolved_mapping[str(resolved_path)] = group

    return resolved_mapping


def load_exclusions(
    exclude_file: str | Path,
    input_directory: str | Path,
) -> dict[str, str]:
    """Load quarantine reasons for files inside the input directory."""
    report_path = Path(exclude_file)
    input_root = Path(input_directory).resolve()

    raw_report = json.loads(report_path.read_text(encoding="utf-8"))

    if not isinstance(raw_report, dict):
        raise TypeError("Exclusion report must be a JSON object.")

    records = raw_report.get("quarantined_documents")

    if not isinstance(records, list):
        raise TypeError(
            "Exclusion report must contain a quarantined_documents list."
        )

    exclusions: dict[str, str] = {}

    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Every quarantine entry must be a JSON object.")

        relative_path = record.get("document")
        reason = record.get("reason")

        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("Every quarantine entry needs a document path.")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"Every quarantine entry needs a reason: {relative_path}"
            )

        candidate = Path(relative_path)

        if candidate.is_absolute():
            raise ValueError(
                f"Quarantine paths must be relative: {relative_path}"
            )

        resolved_path = (input_root / candidate).resolve()

        if not resolved_path.is_relative_to(input_root):
            raise ValueError(
                f"Quarantine path escapes the input directory: {relative_path}"
            )

        if not resolved_path.is_file():
            raise ValueError(
                f"Quarantined document does not exist: {relative_path}"
            )

        key = str(resolved_path)

        if key in exclusions:
            raise ValueError(
                f"Duplicate quarantine entry: {relative_path}"
            )

        exclusions[key] = reason.strip()

    return exclusions


def _reject_symlink_components(path: Path, description: str) -> None:
    """Reject symlinks and junctions in every existing path component."""
    candidate = Path(path).expanduser()

    if ".." in candidate.parts:
        raise ValueError(f"{description} must not contain '..' path components")

    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    current = Path(candidate.anchor)

    for part in candidate.parts[1:]:
        current = current / part
        is_junction = getattr(current, "is_junction", lambda: False)()
        if current.is_symlink() or is_junction:
            raise ValueError(
                f"{description} cannot traverse a symbolic link or junction"
            )


def run_training(args, config, logger) -> None:
    """Run guarded single-process training from prepared JSONL files."""
    if config.curriculum is not None:
        raise ValueError(
            "This trainer does not yet implement curriculum progression. "
            "Use a non-curriculum configuration; the V2 configuration "
            "has not been modified."
        )

    if config.training.num_workers != 0:
        raise ValueError(
            "The initial training CLI requires training.num_workers=0. "
            "Set it to 0 in the selected configuration until multi-worker "
            "streaming has dedicated integration tests."
        )

    if (
        args.max_validation_batches is not None
        and args.max_validation_batches < 1
    ):
        raise ValueError("--max-validation-batches must be positive")

    dataset_directory = Path(args.dataset).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    checkpoint_argument = Path(args.checkpoint_dir).expanduser()
    _reject_symlink_components(
        checkpoint_argument,
        "Checkpoint directory",
    )
    checkpoint_directory = checkpoint_argument.resolve()
    train_path = dataset_directory / "train.jsonl"
    validation_path = dataset_directory / "validation.jsonl"

    if not dataset_directory.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: {dataset_directory}"
        )

    for required_file in (train_path, validation_path):
        if not required_file.is_file():
            raise FileNotFoundError(
                f"Required dataset split does not exist: {required_file}"
            )

    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file does not exist: {tokenizer_path}"
        )

    if (
        checkpoint_directory == dataset_directory
        or dataset_directory in checkpoint_directory.parents
    ):
        raise ValueError(
            "Checkpoint directory must not be inside the dataset directory"
        )

    if tokenizer_path.is_relative_to(checkpoint_directory):
        raise ValueError(
            "Checkpoint directory must not contain the tokenizer artifact"
        )

    resume_argument = (
        Path(args.resume).expanduser()
        if args.resume is not None
        else None
    )

    if resume_argument is not None:
        _reject_symlink_components(
            resume_argument,
            "Resume checkpoint",
        )

    resume_path = (
        resume_argument.resolve()
        if resume_argument is not None
        else None
    )

    if resume_path is not None:
        if resume_path.name not in {"latest.pt", "best.pt"}:
            raise ValueError(
                "--resume must reference latest.pt or best.pt"
            )
        if resume_path.parent != checkpoint_directory:
            raise ValueError(
                "--resume must point to a checkpoint directly inside "
                "--checkpoint-dir"
            )
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_path}"
            )

    if checkpoint_directory.exists():
        if not checkpoint_directory.is_dir():
            raise NotADirectoryError(
                f"Checkpoint path is not a directory: {checkpoint_directory}"
            )

        for checkpoint_name in ("latest.pt", "best.pt"):
            _reject_symlink_components(
                checkpoint_directory / checkpoint_name,
                f"Checkpoint file {checkpoint_name}",
            )

        existing_entries = list(checkpoint_directory.iterdir())

        if existing_entries and resume_path is None:
            raise FileExistsError(
                "Checkpoint directory is not empty. Choose a new directory "
                "or explicitly pass --resume with a checkpoint in it: "
                f"{checkpoint_directory}"
            )

        if resume_path is not None:
            unexpected = [
                entry
                for entry in existing_entries
                if entry not in {
                    resume_path,
                    checkpoint_directory / "latest.pt",
                    checkpoint_directory / "best.pt",
                }
            ]
            if unexpected:
                raise FileExistsError(
                    "Checkpoint directory contains unexpected files: "
                    + ", ".join(str(item) for item in unexpected)
                )

    # Load optional training dependencies only for the train command.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from vyrex_ai.data.dataset import TokenBlockDataset
    from vyrex_ai.models import VyrexConfig, VyrexTransformer
    from vyrex_ai.trainer import train_model

    tokenizer = VyrexTokenizer.load(tokenizer_path)

    if tokenizer.vocab_size > config.model.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary exceeds model vocabulary: "
            f"{tokenizer.vocab_size} > {config.model.vocab_size}"
        )

    train_dataset = TokenBlockDataset(
        train_path,
        tokenizer,
        config.model.max_seq_len,
    )
    validation_dataset = TokenBlockDataset(
        validation_path,
        tokenizer,
        config.model.max_seq_len,
    )

    # Fail before creating a model or checkpoint directory if either split
    # cannot produce a complete causal-LM token block.
    if next(iter(train_dataset), None) is None:
        raise ValueError(
            "Training split contains insufficient tokens for one complete "
            f"{config.model.max_seq_len + 1}-token block"
        )

    if next(iter(validation_dataset), None) is None:
        raise ValueError(
            "Validation split contains insufficient tokens for one complete "
            f"{config.model.max_seq_len + 1}-token block"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.micro_batch_size,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.micro_batch_size,
        num_workers=0,
        drop_last=False,
    )

    random.seed(config.project.get("seed", 42))
    np.random.seed(config.project.get("seed", 42))
    torch.manual_seed(config.project.get("seed", 42))

    model_config = VyrexConfig(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        tie_embeddings=config.model.tie_embeddings,
    )
    model = VyrexTransformer(model_config)

    logger.info("Starting VYREX AI training")
    logger.info("Dataset: %s", dataset_directory)
    logger.info("Tokenizer: %s", tokenizer_path)
    logger.info("Checkpoint directory: %s", checkpoint_directory)

    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        training_config=config.training,
        checkpoint_dir=checkpoint_directory,
        device=args.device,
        resume_from=resume_path,
        max_validation_batches=args.max_validation_batches,
        metrics_path=f"{config.paths['logs_dir']}/training_metrics.csv",
    )

    print("VYREX AI TRAINING COMPLETE")
    print(f"Completed steps: {result['completed_steps']}")
    print(f"Device: {result['device']}")
    print(f"Precision: {result['precision']}")
    print(f"Best validation loss: {result['best_validation_loss']}")
    print(f"Checkpoint directory: {checkpoint_directory}")
    print("Training and validation source files were not modified.")



def run_curriculum_training(args, config, logger) -> None:
    """Run guarded five-stage curriculum training."""
    import hashlib
    from dataclasses import asdict

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from vyrex_ai.config import EXPECTED_CURRICULUM_STAGES
    from vyrex_ai.curriculum_trainer import train_curriculum
    from vyrex_ai.data.dataset import TokenBlockDataset
    from vyrex_ai.models import VyrexConfig, VyrexTransformer

    if config.curriculum is None:
        raise ValueError(
            "The selected configuration does not define a curriculum."
        )

    if tuple(config.curriculum.stages) != tuple(EXPECTED_CURRICULUM_STAGES):
        raise ValueError(
            "Configuration curriculum stages do not match the required "
            "five-stage order."
        )

    if not config.curriculum.progressive_training:
        raise ValueError("Curriculum progressive_training must be enabled.")
    if not config.curriculum.evaluate_before_advancing:
        raise ValueError(
            "Curriculum evaluate_before_advancing must be enabled."
        )
    if not config.curriculum.retain_previous_stage_knowledge:
        raise ValueError(
            "Curriculum retain_previous_stage_knowledge must be enabled."
        )

    if args.num_workers != 0:
        raise ValueError("Curriculum training currently requires --num-workers 0.")

    if (
        args.max_validation_batches is not None
        and args.max_validation_batches < 1
    ):
        raise ValueError("--max-validation-batches must be positive")

    dataset_argument = Path(args.dataset).expanduser()
    tokenizer_argument = Path(args.tokenizer).expanduser()
    checkpoint_argument = Path(args.checkpoint_dir).expanduser()

    _reject_symlink_components(dataset_argument, "Curriculum dataset")
    _reject_symlink_components(tokenizer_argument, "Tokenizer")
    _reject_symlink_components(checkpoint_argument, "Checkpoint directory")

    dataset_directory = dataset_argument.resolve()
    tokenizer_path = tokenizer_argument.resolve()
    checkpoint_directory = checkpoint_argument.resolve()

    if not dataset_directory.is_dir():
        raise NotADirectoryError(
            f"Curriculum dataset directory does not exist: {dataset_directory}"
        )
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file does not exist: {tokenizer_path}"
        )

    if (
        checkpoint_directory == dataset_directory
        or dataset_directory in checkpoint_directory.parents
        or checkpoint_directory in dataset_directory.parents
    ):
        raise ValueError(
            "Checkpoint and curriculum dataset directories must not contain "
            "one another."
        )

    if tokenizer_path.is_relative_to(checkpoint_directory):
        raise ValueError(
            "Checkpoint directory must not contain the tokenizer artifact."
        )

    resume_argument = (
        Path(args.resume).expanduser()
        if args.resume is not None
        else None
    )
    if resume_argument is not None:
        _reject_symlink_components(resume_argument, "Resume checkpoint")

    resume_path = (
        resume_argument.resolve()
        if resume_argument is not None
        else None
    )

    if resume_path is not None:
        if resume_path.name != "latest.pt":
            raise ValueError(
                "--resume must reference latest.pt; best.pt is not resumable."
            )
        if resume_path.parent != checkpoint_directory:
            raise ValueError(
                "--resume must point directly inside --checkpoint-dir."
            )
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_path}"
            )

    if checkpoint_directory.exists():
        if not checkpoint_directory.is_dir():
            raise NotADirectoryError(
                f"Checkpoint path is not a directory: {checkpoint_directory}"
            )

        for name in ("latest.pt", "best.pt"):
            _reject_symlink_components(
                checkpoint_directory / name,
                f"Checkpoint file {name}",
            )

        entries = list(checkpoint_directory.iterdir())

        if entries and resume_path is None:
            raise FileExistsError(
                "Checkpoint directory is not empty. Use a new directory "
                "or explicitly resume from its latest.pt."
            )

        if resume_path is not None:
            allowed = {
                checkpoint_directory / "latest.pt",
                checkpoint_directory / "best.pt",
            }
            unexpected = [item for item in entries if item not in allowed]
            if unexpected:
                raise FileExistsError(
                    "Checkpoint directory contains unexpected entries: "
                    + ", ".join(str(item) for item in unexpected)
                )

    tokenizer = VyrexTokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > config.model.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary exceeds model vocabulary: "
            f"{tokenizer.vocab_size} > {config.model.vocab_size}"
        )

    stage_loaders = {}
    stage_file_hashes = {}

    def hash_file(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for stage in EXPECTED_CURRICULUM_STAGES:
        stage_directory = dataset_directory / stage
        _reject_symlink_components(
            stage_directory,
            f"Curriculum stage directory {stage}",
        )

        if not stage_directory.is_dir():
            raise FileNotFoundError(
                f"Curriculum stage directory is missing: {stage_directory}"
            )

        required_files = {
            split: stage_directory / f"{split}.jsonl"
            for split in ("train", "validation", "test")
        }
        required_files["manifest"] = (
            stage_directory / "curriculum_manifest.jsonl"
        )

        for label, file_path in required_files.items():
            _reject_symlink_components(
                file_path,
                f"{stage} {label} file",
            )
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Required {stage} {label} file is missing: {file_path}"
                )

        train_dataset = TokenBlockDataset(
            required_files["train"],
            tokenizer,
            config.model.max_seq_len,
        )
        validation_dataset = TokenBlockDataset(
            required_files["validation"],
            tokenizer,
            config.model.max_seq_len,
        )

        if next(iter(train_dataset), None) is None:
            raise ValueError(
                f"{stage} training split cannot produce a complete "
                f"{config.model.max_seq_len + 1}-token block."
            )

        if next(iter(validation_dataset), None) is None:
            raise ValueError(
                f"{stage} validation split cannot produce a complete "
                f"{config.model.max_seq_len + 1}-token block."
            )

        stage_loaders[stage] = (
            DataLoader(
                train_dataset,
                batch_size=config.training.micro_batch_size,
                num_workers=0,
                drop_last=False,
            ),
            DataLoader(
                validation_dataset,
                batch_size=config.training.micro_batch_size,
                num_workers=0,
                drop_last=False,
            ),
        )

        stage_file_hashes[stage] = {
            label: hash_file(file_path)
            for label, file_path in required_files.items()
        }

    def hash_json(value) -> str:
        encoded = __import__("json").dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    resume_signature = {
        "curriculum_stages": list(EXPECTED_CURRICULUM_STAGES),
        "stage_files": stage_file_hashes,
        "tokenizer_sha256": hash_file(tokenizer_path),
        "model_config_sha256": hash_json(asdict(config.model)),
        "training_config_sha256": hash_json(asdict(config.training)),
        "curriculum_config_sha256": hash_json(asdict(config.curriculum)),
        "sequence_length": config.model.max_seq_len,
        "micro_batch_size": config.training.micro_batch_size,
        "gradient_accumulation_steps": (
            config.training.gradient_accumulation_steps
        ),
        "num_workers": 0,
    }

    seed = config.project.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_config = VyrexConfig(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        tie_embeddings=config.model.tie_embeddings,
    )
    model = VyrexTransformer(model_config)

    logger.info("Starting VYREX AI five-stage curriculum training")
    logger.info("Curriculum dataset: %s", dataset_directory)
    logger.info("Tokenizer: %s", tokenizer_path)
    logger.info("Checkpoint directory: %s", checkpoint_directory)

    result = train_curriculum(
        model=model,
        stage_loaders=stage_loaders,
        training_config=config.training,
        checkpoint_dir=checkpoint_directory,
        resume_signature=resume_signature,
        device=args.device,
        resume_from=resume_path,
        max_validation_batches=args.max_validation_batches,
    )

    print("VYREX AI CURRICULUM TRAINING COMPLETE")
    print(f"Completed optimizer steps: {result['completed_steps']}")
    print(f"Completed stages: {', '.join(result['completed_stages'])}")
    print(f"Device: {result['device']}")
    print(f"Precision: {result['precision']}")
    print(f"Best validation loss: {result['best_validation_loss']}")
    print(f"Latest checkpoint: {result['checkpoint']}")
    print("Curriculum source files were not modified.")


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "doctor":
        report = inspect_runtime()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_runtime_report(report)
        return

    if args.command == "train-tokenizer":
        if not args.dataset_name.strip():
            raise ValueError("--dataset-name cannot be empty")

        dataset_directory = Path(args.dataset).resolve()
        output_directory = Path(args.output).resolve()

        corpus_stats = inspect_training_corpus(dataset_directory)

        tokenizer = VyrexTokenizer.train(
            iter_training_texts(dataset_directory),
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )

        saved_tokenizer, saved_metadata = save_tokenizer_artifacts(
            tokenizer,
            corpus_stats,
            output_directory,
            dataset_name=args.dataset_name,
            overwrite=args.overwrite,
        )
        print("VYREX TOKENIZER TRAINING COMPLETE")
        print(f"Dataset: {args.dataset}")
        print(f"Training documents: {corpus_stats.document_count}")
        print(f"Corpus SHA-256: {corpus_stats.sha256}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Tokenizer: {saved_tokenizer}")
        print(f"Metadata: {saved_metadata}")
        print("No source dataset files were modified.")
        return

    if args.command == "prepare-data":
        input_directory = Path(args.input).resolve()
        source_groups = load_source_groups(args.groups, input_directory)
        exclusions = (
            load_exclusions(args.exclude_list, input_directory)
            if args.exclude_list
            else None
        )

        report = prepare_dataset(
            input_directory=input_directory,
            source_groups=source_groups,
            output_ratios=(
                args.train_ratio,
                args.validation_ratio,
                args.test_ratio,
            ),
            seed=args.seed,
            exclusions=exclusions,
        )

        result = export_dataset(
            report=report,
            source_groups=source_groups,
            output_directory=args.output,
            corpus_root=input_directory,
        )

        print("VYREX DATA PREPARATION COMPLETE")
        print(f"Documents read: {report.total_read}")
        print(f"Accepted: {report.accepted_count}")
        print(f"Rejected: {report.rejected_count}")
        print(f"Quarantined: {len(report.quarantined_documents)}")
        print(f"Duplicates: {report.duplicate_count}")
        print(f"Read failures: {len(report.read_failures)}")
        print(f"Training records: {result.train_count}")
        print(f"Validation records: {result.validation_count}")
        print(f"Test records: {result.test_count}")
        print(f"Output directory: {result.output_directory}")
        print(f"Audit manifest: {result.manifest_path}")
        print("No original source documents were modified.")
        return

    if args.command == "evaluate-english-levels":
        from vyrex_ai.data.english_level_ml_experiment import (
            run_ml_experiment,
        )

        result = run_ml_experiment(
            dataset_path=args.dataset,
            report_path=args.report,
            n_splits=args.folds,
            random_state=args.seed,
        )

        print("VYREX ENGLISH-LEVEL ML EXPERIMENT COMPLETE")
        print(json.dumps(result, indent=2))
        print("Full metrics and out-of-fold predictions saved in the report.")
        print("No production model was approved or deployed.")
        return

    if args.command == "evaluate-english-level-v2-3":
        from vyrex_ai.data.english_level_v2_3_experiment import (
            run_v2_3_experiment,
        )

        result = run_v2_3_experiment(
            dataset_path=args.dataset,
            report_path=args.report,
            n_splits=args.folds,
            random_state=args.seed,
        )

        print("VYREX V2.3 FEATURE VALIDATION COMPLETE")
        print(json.dumps(result, indent=2))
        print("No production model was approved or deployed.")
        return

    if args.command == "evaluate-english-level-diagnostics":
        from vyrex_ai.data.english_level_diagnostics import (
            run_diagnostic_evaluation,
        )

        result = run_diagnostic_evaluation(
            dataset_path=args.dataset,
            report_path=args.report,
            records_path=args.records,
            maximum_records=args.maximum_records,
        )

        print("VYREX V2 VS V2.1 DIAGNOSTIC EVALUATION COMPLETE")
        print(json.dumps(result, indent=2))
        print("No human labels or production approvals were created.")
        return

    if args.command == "audit-english-level-v2-2":
        from vyrex_ai.data.english_level_v2_2_audit import (
            run_v2_2_quality_audit,
        )

        result = run_v2_2_quality_audit(
            dataset_path=args.dataset,
            report_path=args.report,
            maximum_records=args.maximum_records,
            seed=args.seed,
            correlation_threshold=args.correlation_threshold,
            reference_dataset_path=args.reference_dataset,
        )

        print("VYREX V2.2 FEATURE QUALITY AUDIT COMPLETE")
        print(json.dumps(result, indent=2))
        print("No human labels or production approvals were created.")
        return

    if args.command == "benchmark-english-level-v2-2-ablation":
        from vyrex_ai.data.english_level_v2_2_ablation_benchmark import (
            run_v2_2_ablation_benchmark,
        )

        result = run_v2_2_ablation_benchmark(
            dataset_path=args.dataset,
            report_path=args.report,
            maximum_records=args.maximum_records,
            seed=args.seed,
            correlation_threshold=args.correlation_threshold,
        )

        print("VYREX V2.2 FEATURE ABLATION BENCHMARK COMPLETE")
        print(json.dumps(result, indent=2))
        print("No human labels or production approvals were created.")
        return

    if args.command == "import-cefr-sp-wikiauto":
        from vyrex_ai.data.english_level_cefr_sp import (
            import_cefr_sp_wikiauto,
        )

        result = import_cefr_sp_wikiauto(
            source_directory=args.source,
            output_directory=args.output,
        )

        print("VYREX CEFR-SP WIKI-AUTO IMPORT COMPLETE")
        print(json.dumps(result, indent=2))
        print("Original source files were not modified.")
        print("Both annotator labels were preserved.")
        print("No model training or production approval occurred.")
        return

    if args.command == "evaluate-english-level-cefr-sp-v2-3":
        from vyrex_ai.data.english_level_cefr_sp_v2_3 import (
            run_cefr_sp_v2_3_experiment,
        )

        result = run_cefr_sp_v2_3_experiment(
            dataset_directory=args.dataset,
            report_path=args.report,
            n_splits=args.folds,
            random_state=args.seed,
        )
        print(json.dumps(result, indent=2))
        print("No production model was approved or deployed.")
        return

    if args.command == "expand-english-levels":
        from vyrex_ai.data.english_level_ml_expansion import (
            expand_tatoeba_candidates,
        )

        if not args.dry_run and (not args.output or not args.manifest):
            raise ValueError(
                "--output and --manifest are required unless --dry-run is used"
            )

        result = expand_tatoeba_candidates(
            input_path=args.input,
            output_path=args.output,
            manifest_path=args.manifest,
            maximum_records=args.maximum_records,
            dry_run=args.dry_run,
        )

        print("VYREX ENGLISH-LEVEL CANDIDATE EXPANSION COMPLETE")
        print(json.dumps(result, indent=2))
        if args.dry_run:
            print("Dry run: no candidate or manifest files were published.")
        else:
            print("Candidate queue and manifest were published.")
        print("No human labels or production model approvals were created.")
        return

    config = load_config(args.config)
    logger = configure_logging(config.paths["logs_dir"])

    if args.command == "train-curriculum":
        run_curriculum_training(args, config, logger)
        return

    if args.command == "train":
        run_training(args, config, logger)
        return

    if args.command == "show-config":
        logger.info("Configuration validated: %s", Path(args.config))
        print(f"Project: {config.project['name']} v{config.project['version']}")
        print(f"Mode/language: {config.project['mode']} / {config.project['language']}")
        print(
            "Model: "
            f"{config.model.num_layers} layers, {config.model.hidden_size} hidden, "
            f"{config.model.num_heads} heads, vocab {config.model.vocab_size}, "
            f"context {config.model.max_seq_len}"
        )
        print(f"Data mixture: {config.data.mixture}")
        print("Status: CONFIGURATION VALID")
        return

    if args.command == "init":
        for key, value in config.paths.items():
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Directory ready: %s=%s", key, path)
        print("VYREX AI Phase 1 directories initialized.")
        print("No dataset was downloaded and no model training was started.")


if __name__ == "__main__":
    main()



