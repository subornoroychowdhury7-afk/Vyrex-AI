
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vyrex_ai.data.corpus import CorpusStats
from vyrex_ai.tokenizer import VyrexTokenizer
from vyrex_ai.tokenizer_artifacts import save_tokenizer_artifacts


@pytest.fixture
def tokenizer() -> VyrexTokenizer:
    return VyrexTokenizer.train(
        ["Hello world.", "VYREX AI tokenizer test."],
        vocab_size=300,
    )


@pytest.fixture
def corpus_stats() -> CorpusStats:
    return CorpusStats(
        document_count=2,
        character_count=36,
        text_bytes=36,
        file_size_bytes=100,
        sha256="a" * 64,
    )


def test_save_tokenizer_artifacts(
    tmp_path: Path,
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
) -> None:
    tokenizer_path, metadata_path = save_tokenizer_artifacts(
        tokenizer,
        corpus_stats,
        tmp_path,
        dataset_name="synthetic-test",
    )

    assert tokenizer_path.is_file()
    assert metadata_path.is_file()

    loaded = VyrexTokenizer.load(tokenizer_path)
    assert loaded.vocab_size == tokenizer.vocab_size

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["training_corpus"]["dataset_name"] == "synthetic-test"


def test_refuses_overwrite(
    tmp_path: Path,
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
) -> None:
    save_tokenizer_artifacts(
        tokenizer, corpus_stats, tmp_path, "synthetic-test"
    )

    with pytest.raises(FileExistsError, match="already exist"):
        save_tokenizer_artifacts(
            tokenizer, corpus_stats, tmp_path, "synthetic-test"
        )


def test_explicit_overwrite(
    tmp_path: Path,
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
) -> None:
    save_tokenizer_artifacts(
        tokenizer, corpus_stats, tmp_path, "synthetic-test"
    )

    tokenizer_path, metadata_path = save_tokenizer_artifacts(
        tokenizer,
        corpus_stats,
        tmp_path,
        "synthetic-test-updated",
        overwrite=True,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["training_corpus"]["dataset_name"] == "synthetic-test-updated"
    assert tokenizer_path.is_file()


@pytest.mark.parametrize(
    "artifact_name",
    ["tokenizer.json", "tokenizer_metadata.json"],
)
def test_rejects_artifact_directory(
    tmp_path: Path,
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    artifact_name: str,
) -> None:
    (tmp_path / artifact_name).mkdir()

    with pytest.raises((IsADirectoryError, FileExistsError)):
        save_tokenizer_artifacts(
            tokenizer,
            corpus_stats,
            tmp_path,
            "synthetic-test",
            overwrite=True,
        )


def test_rejects_output_path_that_is_a_file(
    tmp_path: Path,
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("protected", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        save_tokenizer_artifacts(
            tokenizer,
            corpus_stats,
            output_file,
            "synthetic-test",
        )


def test_staging_failure_preserves_existing_artifacts(
    tmp_path: Path,
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
) -> None:
    tokenizer_path, metadata_path = save_tokenizer_artifacts(
        tokenizer, corpus_stats, tmp_path, "original"
    )

    original_tokenizer = tokenizer_path.read_bytes()
    original_metadata = metadata_path.read_bytes()

    with pytest.raises(FileExistsError):
        save_tokenizer_artifacts(
            tokenizer,
            corpus_stats,
            tmp_path,
            "replacement",
        )

    assert tokenizer_path.read_bytes() == original_tokenizer
    assert metadata_path.read_bytes() == original_metadata


def test_temporary_directory_is_cleaned(
    tmp_path: Path,
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
) -> None:
    save_tokenizer_artifacts(
        tokenizer,
        corpus_stats,
        tmp_path,
        "synthetic-test",
    )

    assert not list(tmp_path.glob(".vyrex-tokenizer-*"))
