from pathlib import Path

import pytest

from vyrex_ai.data.readers import (
    read_directory,
    read_document,
)


def test_read_document(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello VYREX AI.", encoding="utf-8")

    document = read_document(file_path)

    assert document.text == "Hello VYREX AI."
    assert document.source_path == str(file_path.resolve())


def test_read_document_removes_utf8_bom(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("\ufeffHello VYREX.", encoding="utf-8")

    document = read_document(file_path)

    assert document.text == "Hello VYREX."


def test_read_directory_reads_supported_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("First document.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Second document", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("not supported", encoding="utf-8")

    result = read_directory(tmp_path)

    assert len(result.documents) == 2
    assert len(result.failures) == 0
    assert [Path(doc.source_path).name for doc in result.documents] == [
        "a.txt",
        "b.md",
    ]


def test_read_directory_reports_unreadable_files(tmp_path: Path):
    file_path = tmp_path / "broken.txt"
    file_path.write_bytes(b"\xff\xfe\xfa")

    result = read_directory(tmp_path)

    assert len(result.documents) == 0
    assert len(result.failures) == 1
    assert "UnicodeDecodeError" in result.failures[0].reason


def test_missing_directory_raises_error(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        read_directory(tmp_path / "missing")
