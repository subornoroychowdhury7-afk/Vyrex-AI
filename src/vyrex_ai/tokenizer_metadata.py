"""Reproducibility metadata for VYREX tokenizer artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from vyrex_ai.data.corpus import CorpusStats
from vyrex_ai.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    TOKENIZER_NAME,
    VyrexTokenizer,
)


def save_tokenizer_metadata(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    directory: str | Path,
    dataset_name: str,
    overwrite: bool = False,
) -> Path:
    """Save tokenizer configuration and training-corpus provenance."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")

    if not isinstance(corpus_stats, CorpusStats):
        raise TypeError("corpus_stats must be a CorpusStats instance")

    if corpus_stats.document_count < 1:
        raise ValueError("corpus_stats must contain at least one document")

    count_fields = (
        corpus_stats.document_count,
        corpus_stats.character_count,
        corpus_stats.text_bytes,
        corpus_stats.file_size_bytes,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in count_fields
    ):
        raise ValueError("corpus statistics counts must be non-negative integers")

    if (
        not isinstance(corpus_stats.sha256, str)
        or len(corpus_stats.sha256) != 64
        or any(character not in "0123456789abcdef" for character in corpus_stats.sha256)
    ):
        raise ValueError("corpus_stats.sha256 must be a 64-character lowercase hexadecimal digest")

    if tokenizer.requested_vocab_size is None or tokenizer.min_frequency is None:
        raise ValueError(
            "Tokenizer training configuration is unknown; "
            "metadata cannot be generated accurately."
        )

    output_dir = Path(directory)

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Tokenizer output directory does not exist: {output_dir}"
        )

    output_path = output_dir / "tokenizer_metadata.json"
    temporary_path = output_dir / "tokenizer_metadata.json.tmp"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Tokenizer metadata already exists: {output_path}. "
            "Pass overwrite=True to replace it intentionally."
        )

    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary metadata file already exists: {temporary_path}"
        )

    metadata = {
        "schema_version": 1,
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "algorithm": "byte-level BPE",
            "requested_vocab_size": tokenizer.requested_vocab_size,
            "actual_vocab_size": tokenizer.vocab_size,
            "min_frequency": tokenizer.min_frequency,
            "special_tokens": {
                PAD_TOKEN: tokenizer.pad_token_id,
                BOS_TOKEN: tokenizer.bos_token_id,
                EOS_TOKEN: tokenizer.eos_token_id,
            },
        },
        "library": {
            "name": "tokenizers",
            "version": version("tokenizers"),
        },
        "training_corpus": {
            "dataset_name": dataset_name.strip(),
            **asdict(corpus_stats),
        },
    }

    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                metadata,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        if overwrite:
            temporary_path.replace(output_path)
        else:
            temporary_path.rename(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path
