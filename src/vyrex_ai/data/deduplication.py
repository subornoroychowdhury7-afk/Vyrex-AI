"""Non-destructive exact-duplicate detection for VYREX AI."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .readers import Document


@dataclass(frozen=True)
class DuplicateRecord:
    source_path: str
    duplicate_of: str
    fingerprint: str


@dataclass(frozen=True)
class DeduplicationResult:
    unique_documents: tuple[Document, ...]
    duplicates: tuple[DuplicateRecord, ...]


def fingerprint_document(document: Document) -> str:
    """Create a normalized SHA-256 fingerprint for document text."""
    normalized = " ".join(document.text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_documents(
    documents: list[Document] | tuple[Document, ...],
) -> DeduplicationResult:
    """Keep the first occurrence of each normalized exact-text duplicate."""
    seen: dict[str, str] = {}
    unique: list[Document] = []
    duplicates: list[DuplicateRecord] = []

    for document in documents:
        fingerprint = fingerprint_document(document)

        if fingerprint in seen:
            duplicates.append(
                DuplicateRecord(
                    source_path=document.source_path,
                    duplicate_of=seen[fingerprint],
                    fingerprint=fingerprint,
                )
            )
            continue

        seen[fingerprint] = document.source_path
        unique.append(document)

    return DeduplicationResult(
        unique_documents=tuple(unique),
        duplicates=tuple(duplicates),
    )
