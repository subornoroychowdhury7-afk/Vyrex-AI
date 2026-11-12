"""Coordinated saving of VYREX tokenizer artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from vyrex_ai.data.corpus import CorpusStats
from vyrex_ai.tokenizer import VyrexTokenizer
from vyrex_ai.tokenizer_metadata import save_tokenizer_metadata


def save_tokenizer_artifacts(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    directory: str | Path,
    dataset_name: str,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save tokenizer and metadata together with rollback protection."""
    output_dir = Path(directory)

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"Tokenizer output path is not a directory: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "tokenizer.json"
    metadata_path = output_dir / "tokenizer_metadata.json"
    destinations = (tokenizer_path, metadata_path)

    for destination in destinations:
        if destination.is_dir():
            raise IsADirectoryError(
                f"Tokenizer artifact path is a directory: {destination}"
            )
        if destination.is_symlink():
            raise ValueError(
                f"Tokenizer artifact path cannot be a symbolic link: "
                f"{destination}"
            )
        if destination.exists() and not destination.is_file():
            raise OSError(
                f"Tokenizer artifact path is not a regular file: {destination}"
            )

    if not overwrite:
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(
                "Tokenizer artifacts already exist: "
                + ", ".join(str(path) for path in existing)
                + ". Pass overwrite=True to replace them intentionally."
            )

    with tempfile.TemporaryDirectory(
        prefix=".vyrex-tokenizer-",
        dir=output_dir,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory) / "staging"
        staging_dir.mkdir()

        staged_tokenizer = staging_dir / "tokenizer.json"
        staged_metadata = staging_dir / "tokenizer_metadata.json"

        tokenizer.save(staging_dir)

        save_tokenizer_metadata(
            tokenizer,
            corpus_stats,
            staging_dir,
            dataset_name=dataset_name,
        )

        VyrexTokenizer.load(staged_tokenizer)

        with staged_metadata.open("r", encoding="utf-8") as handle:
            json.load(handle)

        staged_files = (staged_tokenizer, staged_metadata)

        if not all(path.is_file() for path in staged_files):
            raise RuntimeError("Tokenizer artifact staging is incomplete.")

        backup_dir = Path(temporary_directory) / "backup"
        backup_dir.mkdir()

        backups: dict[Path, Path] = {}
        published: list[Path] = []

        try:
            for destination in destinations:
                if destination.exists():
                    backup = backup_dir / destination.name
                    shutil.copy2(destination, backup)
                    backups[destination] = backup

            for staged, destination in zip(
                staged_files,
                destinations,
                strict=True,
            ):
                staged.replace(destination)
                published.append(destination)

        except Exception as original_error:
            rollback_errors: list[Exception] = []

            for destination in reversed(published):
                try:
                    backup = backups.get(destination)

                    if backup is not None and backup.exists():
                        backup.replace(destination)
                    elif destination.exists():
                        destination.unlink()
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)

            if rollback_errors:
                details = "; ".join(
                    f"{type(error).__name__}: {error}"
                    for error in rollback_errors
                )
                raise RuntimeError(
                    "Tokenizer publishing failed and rollback also failed: "
                    + details
                ) from original_error

            raise

    return tokenizer_path, metadata_path
