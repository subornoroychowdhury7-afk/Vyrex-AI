from __future__ import annotations

import json
from pathlib import Path

import pytest

from vyrex_ai.data.corpus import CorpusStats
from vyrex_ai.tokenizer import VyrexTokenizer
from vyrex_ai.tokenizer_metadata import save_tokenizer_metadata


@pytest.fixture
def tokenizer() -> VyrexTokenizer:
    texts = [
        "VYREX AI learns English.",
        "A tokenizer converts text into tokens.",
    ] * 20
    return VyrexTokenizer.train(texts, vocab_size=512, min_frequency=1)


@pytest.fixture
def corpus_stats() -> CorpusStats:
    return CorpusStats(
        document_count=40,
        character_count=920,
        text_bytes=920,
        file_size_bytes=1200,
        sha256="a" * 64,
    )


def test_save_tokenizer_metadata(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
) -> None:
    tokenizer.save(tmp_path)
    metadata_path = save_tokenizer_metadata(
        tokenizer, corpus_stats, tmp_path, dataset_name="test-english-corpus"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == 1
    assert metadata["tokenizer"]["name"] == "vyrex-bpe-en-v1"
    assert metadata["tokenizer"]["algorithm"] == "byte-level BPE"
    assert metadata["tokenizer"]["requested_vocab_size"] == 512
    assert metadata["tokenizer"]["actual_vocab_size"] == tokenizer.vocab_size
    assert metadata["tokenizer"]["min_frequency"] == 1
    assert metadata["tokenizer"]["special_tokens"] == {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
    }
    assert metadata["library"]["name"] == "tokenizers"
    assert metadata["training_corpus"]["dataset_name"] == "test-english-corpus"
    assert metadata["training_corpus"]["sha256"] == "a" * 64
    assert metadata["training_corpus"]["document_count"] == 40


@pytest.mark.parametrize("dataset_name", ["", " ", "\t"])
def test_rejects_empty_dataset_name(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
    dataset_name: str,
) -> None:
    with pytest.raises(ValueError, match="dataset_name"):
        save_tokenizer_metadata(
            tokenizer, corpus_stats, tmp_path, dataset_name=dataset_name
        )


def test_rejects_unknown_training_configuration(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
) -> None:
    loaded = VyrexTokenizer.load(tokenizer.save(tmp_path))

    with pytest.raises(ValueError, match="configuration is unknown"):
        save_tokenizer_metadata(
            loaded, corpus_stats, tmp_path, dataset_name="test-corpus"
        )


def test_refuses_to_overwrite_metadata(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
) -> None:
    first_path = save_tokenizer_metadata(
        tokenizer, corpus_stats, tmp_path, dataset_name="first-corpus"
    )
    original_content = first_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        save_tokenizer_metadata(
            tokenizer, corpus_stats, tmp_path, dataset_name="second-corpus"
        )

    assert first_path.read_bytes() == original_content
    assert not (tmp_path / "tokenizer_metadata.json.tmp").exists()


def test_allows_explicit_metadata_overwrite(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
) -> None:
    metadata_path = save_tokenizer_metadata(
        tokenizer, corpus_stats, tmp_path, dataset_name="first-corpus"
    )

    replacement_path = save_tokenizer_metadata(
        tokenizer,
        corpus_stats,
        tmp_path,
        dataset_name="second-corpus",
        overwrite=True,
    )

    metadata = json.loads(replacement_path.read_text(encoding="utf-8"))

    assert replacement_path == metadata_path
    assert metadata["training_corpus"]["dataset_name"] == "second-corpus"
    assert not (tmp_path / "tokenizer_metadata.json.tmp").exists()


def test_requires_existing_output_directory(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
) -> None:
    with pytest.raises(NotADirectoryError):
        save_tokenizer_metadata(
            tokenizer,
            corpus_stats,
            tmp_path / "missing",
            dataset_name="test-corpus",
        )



def test_rejects_invalid_corpus_stats_type(
    tokenizer: VyrexTokenizer,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="CorpusStats"):
        save_tokenizer_metadata(
            tokenizer, {}, tmp_path, dataset_name="test-corpus"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_count", 0),
        ("character_count", -1),
        ("text_bytes", -1),
        ("file_size_bytes", -1),
        ("document_count", True),
    ],
)
def test_rejects_invalid_corpus_counts(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    from dataclasses import replace

    invalid_stats = replace(corpus_stats, **{field: value})

    with pytest.raises(ValueError, match="corpus"):
        save_tokenizer_metadata(
            tokenizer, invalid_stats, tmp_path, dataset_name="test-corpus"
        )


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "A" * 64])
def test_rejects_invalid_corpus_checksum(
    tokenizer: VyrexTokenizer,
    corpus_stats: CorpusStats,
    tmp_path: Path,
    digest: str,
) -> None:
    from dataclasses import replace

    invalid_stats = replace(corpus_stats, sha256=digest)

    with pytest.raises(ValueError, match="sha256"):
        save_tokenizer_metadata(
            tokenizer, invalid_stats, tmp_path, dataset_name="test-corpus"
        )
