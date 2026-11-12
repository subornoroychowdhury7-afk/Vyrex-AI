from pathlib import Path

import pytest

from vyrex_ai.data.corpus import iter_training_texts


def write_dataset(
    directory: Path,
    training_records: list[str],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    content = "".join(
        f'{{"text": {__import__("json").dumps(text)}}}\n'
        for text in training_records
    )
    (directory / "train.jsonl").write_text(content, encoding="utf-8")


def test_reads_training_texts(tmp_path: Path) -> None:
    write_dataset(tmp_path, ["First document.", "Second document."])

    assert list(iter_training_texts(tmp_path)) == [
        "First document.",
        "Second document.",
    ]


def test_reads_only_training_split(tmp_path: Path) -> None:
    write_dataset(tmp_path, ["Training document."])
    (tmp_path / "validation.jsonl").write_text("invalid", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text("invalid", encoding="utf-8")

    assert list(iter_training_texts(tmp_path)) == ["Training document."]


def test_iteration_is_lazy(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(
        '{"text": "Valid document."}\ninvalid json\n',
        encoding="utf-8",
    )

    texts = iter_training_texts(tmp_path)

    assert next(texts) == "Valid document."

    with pytest.raises(ValueError, match="Invalid JSON"):
        next(texts)


def test_missing_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list(iter_training_texts(tmp_path / "missing"))


def test_missing_training_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_training_texts(tmp_path))


def test_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        list(iter_training_texts(tmp_path))


def test_non_object_record(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('"just a string"\n', encoding="utf-8")

    with pytest.raises(TypeError, match="JSON object"):
        list(iter_training_texts(tmp_path))


@pytest.mark.parametrize(
    "record",
    [
        "{}",
        '{"text": 123}',
        '{"text": null}',
        '{"text": ["not", "a", "string"]}',
    ],
)
def test_invalid_text_field(tmp_path: Path, record: str) -> None:
    (tmp_path / "train.jsonl").write_text(record + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match="string 'text' field"):
        list(iter_training_texts(tmp_path))


@pytest.mark.parametrize("text", ["", " ", "\n", "\t"])
def test_empty_document_rejected(tmp_path: Path, text: str) -> None:
    import json

    record = json.dumps({"text": text})
    (tmp_path / "train.jsonl").write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Empty training document"):
        list(iter_training_texts(tmp_path))



def test_inspect_training_corpus(tmp_path: Path) -> None:
    import hashlib

    from vyrex_ai.data.corpus import inspect_training_corpus

    unicode_text = "Caf\u00e9 \U0001F60A"
    write_dataset(tmp_path, ["Hello.", unicode_text])

    train_path = tmp_path / "train.jsonl"
    raw_bytes = train_path.read_bytes()

    stats = inspect_training_corpus(tmp_path)

    assert stats.document_count == 2
    assert stats.character_count == len("Hello.") + len(unicode_text)
    assert stats.text_bytes == len(b"Hello.") + len(
        unicode_text.encode("utf-8")
    )
    assert stats.file_size_bytes == len(raw_bytes)
    assert stats.sha256 == hashlib.sha256(raw_bytes).hexdigest()

def test_inspect_training_corpus_uses_only_train_split(
    tmp_path: Path,
) -> None:
    from vyrex_ai.data.corpus import inspect_training_corpus

    write_dataset(tmp_path, ["Training document."])
    (tmp_path / "validation.jsonl").write_text(
        "invalid JSON",
        encoding="utf-8",
    )

    stats = inspect_training_corpus(tmp_path)

    assert stats.document_count == 1


def test_inspect_training_corpus_rejects_invalid_training_data(
    tmp_path: Path,
) -> None:
    from vyrex_ai.data.corpus import inspect_training_corpus

    (tmp_path / "train.jsonl").write_text(
        "invalid JSON",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        inspect_training_corpus(tmp_path)



def test_inspect_training_corpus_rejects_empty_file(
    tmp_path: Path,
) -> None:
    from vyrex_ai.data.corpus import inspect_training_corpus

    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no documents"):
        inspect_training_corpus(tmp_path)
