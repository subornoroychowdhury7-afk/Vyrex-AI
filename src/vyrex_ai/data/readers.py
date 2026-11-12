"""Safe document readers for VYREX AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    """A source document and its original location."""

    source_path: str
    text: str


@dataclass(frozen=True)
class ReadFailure:
    """A file that could not be read."""

    source_path: str
    reason: str


@dataclass(frozen=True)
class ReadResult:
    """Documents successfully read and files rejected during reading."""

    documents: tuple[Document, ...]
    failures: tuple[ReadFailure, ...]


def read_document(path: str | Path) -> Document:
    """Read one supported text document without modifying its contents."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document extension: {file_path.suffix}")

    text = file_path.read_text(encoding="utf-8-sig")

    return Document(
        source_path=str(file_path.resolve()),
        text=text,
    )


def read_directory(
    directory: str | Path,
    recursive: bool = True,
) -> ReadResult:
    """Read supported files from a directory in deterministic order."""
    root = Path(directory)

    if not root.is_dir():
        raise NotADirectoryError(f"Data directory not found: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()

    files = sorted(
        (
            path for path in iterator
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(root)).casefold(),
    )

    documents: list[Document] = []
    failures: list[ReadFailure] = []

    for file_path in files:
        try:
            documents.append(read_document(file_path))
        except (OSError, UnicodeError, ValueError) as exc:
            failures.append(
                ReadFailure(
                    source_path=str(file_path.resolve()),
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )

    return ReadResult(
        documents=tuple(documents),
        failures=tuple(failures),
    )
