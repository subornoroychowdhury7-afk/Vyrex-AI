"""Command-line interface for VYREX AI project tools."""

from __future__ import annotations

import argparse
import json
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

    config = load_config(args.config)
    logger = configure_logging(config.paths["logs_dir"])

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
