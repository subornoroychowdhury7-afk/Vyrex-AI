from pathlib import Path

import pytest

from vyrex_ai.tokenizer import (
    BOS_TOKEN,
    DEFAULT_VOCAB_SIZE,
    EOS_TOKEN,
    PAD_TOKEN,
    VyrexTokenizer,
)


@pytest.fixture
def tokenizer() -> VyrexTokenizer:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "English language models learn patterns from text.",
        "VYREX AI is an experimental language model.",
        "Punctuation matters: commas, periods, and apostrophes.",
        "Numbers such as 12345 must be preserved.",
        "The quick brown fox learns English.",
    ] * 20

    return VyrexTokenizer.train(
        texts,
        vocab_size=512,
        min_frequency=1,
    )


def test_default_vocabulary_size() -> None:
    assert DEFAULT_VOCAB_SIZE == 16_000


def test_training_configuration_is_retained(
    tokenizer: VyrexTokenizer,
) -> None:
    assert tokenizer.requested_vocab_size == 512
    assert tokenizer.min_frequency == 1


def test_loaded_tokenizer_has_unknown_training_configuration(
    tokenizer: VyrexTokenizer,
    tmp_path: Path,
) -> None:
    saved_path = tokenizer.save(tmp_path)

    loaded = VyrexTokenizer.load(saved_path)

    assert loaded.requested_vocab_size is None
    assert loaded.min_frequency is None


def test_special_token_ids(tokenizer: VyrexTokenizer) -> None:
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2

    assert tokenizer.backend.token_to_id(PAD_TOKEN) == 0
    assert tokenizer.backend.token_to_id(BOS_TOKEN) == 1
    assert tokenizer.backend.token_to_id(EOS_TOKEN) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "English language learning.",
        "Numbers: 1234567890",
        "Punctuation: !?.,;:'\"()[]{}",
        "Unicode: café, naïve, résumé, Ελληνικά, 中文",
        "Whitespace:\nNew line\tand tab.",
        "",
    ],
)
def test_encode_decode_round_trip(
    tokenizer: VyrexTokenizer,
    text: str,
) -> None:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text


def test_encoding_returns_integer_ids(tokenizer: VyrexTokenizer) -> None:
    token_ids = tokenizer.encode("VYREX AI")

    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_save_and_load(
    tokenizer: VyrexTokenizer,
    tmp_path: Path,
) -> None:
    saved_path = tokenizer.save(tmp_path)

    assert saved_path.is_file()

    loaded = VyrexTokenizer.load(saved_path)
    text = "VYREX AI preserves English text."

    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.vocab_size == tokenizer.vocab_size


def test_empty_training_input() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        VyrexTokenizer.train([])


def test_invalid_vocabulary_size() -> None:
    with pytest.raises(ValueError, match="at least 259"):
        VyrexTokenizer.train(["English text"], vocab_size=258)


def test_invalid_min_frequency() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        VyrexTokenizer.train(["English text"], min_frequency=0)


@pytest.mark.parametrize("value", [True, 300.5, "300"])
def test_vocabulary_size_must_be_integer(value) -> None:
    with pytest.raises(TypeError, match="vocab_size must be an integer"):
        VyrexTokenizer.train(["English text"], vocab_size=value)


@pytest.mark.parametrize("value", [False, 1.5, "2"])
def test_min_frequency_must_be_integer(value) -> None:
    with pytest.raises(TypeError, match="min_frequency must be an integer"):
        VyrexTokenizer.train(["English text"], min_frequency=value)


def test_non_string_training_document() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        VyrexTokenizer.train(["English text", 123])  # type: ignore[list-item]


def test_non_string_encode_input(
    tokenizer: VyrexTokenizer,
) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        tokenizer.encode(123)  # type: ignore[arg-type]


def test_missing_tokenizer_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VyrexTokenizer.load(tmp_path / "missing.json")

def test_default_encoding_adds_bos_and_eos(
    tokenizer: VyrexTokenizer,
) -> None:
    token_ids = tokenizer.encode("Hello, VYREX!")

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id


def test_encoding_can_disable_special_tokens(
    tokenizer: VyrexTokenizer,
) -> None:
    token_ids = tokenizer.encode(
        "Hello, VYREX!",
        add_special_tokens=False,
    )

    assert tokenizer.bos_token_id not in token_ids
    assert tokenizer.eos_token_id not in token_ids


def test_save_refuses_to_overwrite(
    tokenizer: VyrexTokenizer,
    tmp_path: Path,
) -> None:
    saved_path = tokenizer.save(tmp_path)
    original_content = saved_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        tokenizer.save(tmp_path)

    assert saved_path.read_bytes() == original_content
    assert not (tmp_path / "tokenizer.json.tmp").exists()


def test_save_allows_explicit_overwrite(
    tokenizer: VyrexTokenizer,
    tmp_path: Path,
) -> None:
    saved_path = tokenizer.save(tmp_path)

    replacement_path = tokenizer.save(tmp_path, overwrite=True)

    assert replacement_path == saved_path
    assert VyrexTokenizer.load(saved_path).vocab_size == tokenizer.vocab_size
    assert not (tmp_path / "tokenizer.json.tmp").exists()


def test_save_cleans_temporary_file_after_failure(
    tokenizer: VyrexTokenizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_path = tmp_path / "tokenizer.json.tmp"

    def fail_save(path: str) -> None:
        Path(path).write_text("incomplete", encoding="utf-8")
        raise OSError("Simulated save failure")

    monkeypatch.setattr(tokenizer.backend, "save", fail_save)

    with pytest.raises(OSError, match="Simulated save failure"):
        tokenizer.save(tmp_path)

    assert not temporary_path.exists()
    assert not (tmp_path / "tokenizer.json").exists()
