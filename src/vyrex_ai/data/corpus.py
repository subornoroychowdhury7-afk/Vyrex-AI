"""Streaming readers for approved VYREX training corpora."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


def iter_training_texts(
    dataset_directory: str | Path,
) -> Iterator[str]:
    """Yield non-empty text records from a dataset's train.jsonl only."""
    directory = Path(dataset_directory)

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: {directory}"
        )

    train_path = directory / "train.jsonl"

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training split does not exist: {train_path}"
        )

    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {train_path}, line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected a JSON object in {train_path}, "
                    f"line {line_number}."
                )

            text = record.get("text")

            if not isinstance(text, str):
                raise TypeError(
                    f"Expected a string 'text' field in {train_path}, "
                    f"line {line_number}."
                )

            if not text.strip():
                raise ValueError(
                    f"Empty training document in {train_path}, "
                    f"line {line_number}."
                )

            yield text


@dataclass(frozen=True)
class CorpusStats:
    """Reproducibility statistics for a training corpus."""

    document_count: int
    character_count: int
    text_bytes: int
    file_size_bytes: int
    sha256: str


def inspect_training_corpus(
    dataset_directory: str | Path,
) -> CorpusStats:
    """Calculate streaming statistics and the exact training-file checksum."""
    import hashlib

    directory = Path(dataset_directory)

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: {directory}"
        )

    train_path = directory / "train.jsonl"

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training split does not exist: {train_path}"
        )

    document_count = 0
    character_count = 0
    text_bytes = 0

    for text in iter_training_texts(directory):
        document_count += 1
        character_count += len(text)
        text_bytes += len(text.encode("utf-8"))

    if document_count == 0:
        raise ValueError(
            f"Training corpus contains no documents: {train_path}"
        )

    digest = hashlib.sha256()

    with train_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return CorpusStats(
        document_count=document_count,
        character_count=character_count,
        text_bytes=text_bytes,
        file_size_bytes=train_path.stat().st_size,
        sha256=digest.hexdigest(),
    )
