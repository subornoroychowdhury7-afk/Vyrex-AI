import json

import pytest
import torch

from vyrex_ai.data.dataset import TokenBlockDataset


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return [1, *[ord(char) + 10 for char in text], 2]


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_fixed_length_blocks(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"text": "ab"}, {"text": "cd"}])

    dataset = TokenBlockDataset(path, FakeTokenizer(), 3)
    blocks = list(dataset)

    assert blocks
    assert all(block.shape == (4,) for block in blocks)
    assert all(block.dtype == torch.long for block in blocks)


def test_document_boundary_is_preserved(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"text": "a"}, {"text": "bc"}])

    dataset = TokenBlockDataset(path, FakeTokenizer(), 3)
    blocks = list(dataset)

    assert len(blocks) == 1
    assert blocks[0].tolist() == [1, 107, 2, 1]


def test_incomplete_final_block_is_discarded(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"text": "a"}])

    dataset = TokenBlockDataset(path, FakeTokenizer(), 3)

    assert list(dataset) == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenBlockDataset(
            tmp_path / "missing.jsonl",
            FakeTokenizer(),
            3,
        )


def test_invalid_sequence_length_raises(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"text": "hello"}])

    with pytest.raises(ValueError):
        TokenBlockDataset(path, FakeTokenizer(), 0)


def test_invalid_jsonl_record_raises(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "hello"}\nnot-json\n', encoding="utf-8")

    dataset = TokenBlockDataset(path, FakeTokenizer(), 3)

    with pytest.raises(ValueError, match="Invalid JSON"):
        list(dataset)


def test_empty_document_raises(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"text": "   "}])

    dataset = TokenBlockDataset(path, FakeTokenizer(), 3)

    with pytest.raises(ValueError, match="non-empty text"):
        list(dataset)


def test_non_object_record_raises(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('"hello"\n', encoding="utf-8")

    dataset = TokenBlockDataset(path, FakeTokenizer(), 3)

    with pytest.raises(TypeError, match="must be an object"):
        list(dataset)
