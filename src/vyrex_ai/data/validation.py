"""Non-destructive document quality checks for VYREX AI."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .readers import Document


@dataclass(frozen=True)
class ValidationIssue:
    """A quality issue found in a document."""

    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome; the source document is never modified."""

    source_path: str
    accepted: bool
    issues: tuple[ValidationIssue, ...]


def validate_document(
    document: Document,
    min_chars: int = 100,
    max_chars: int = 2_000_000,
) -> ValidationResult:
    """Check document quality and report issues without repairing text."""
    if min_chars < 0:
        raise ValueError("min_chars cannot be negative")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if min_chars > max_chars:
        raise ValueError("min_chars cannot exceed max_chars")

    text = document.text
    issues: list[ValidationIssue] = []
    accepted = True

    if not text.strip():
        issues.append(
            ValidationIssue("empty_document", "error", "Document is empty.")
        )
        accepted = False
    elif len(text.strip()) < min_chars:
        issues.append(
            ValidationIssue(
                "very_short_document",
                "warning",
                f"Document contains fewer than {min_chars} non-whitespace characters.",
            )
        )

    if len(text) > max_chars:
        issues.append(
            ValidationIssue(
                "document_too_long",
                "error",
                f"Document exceeds the {max_chars}-character limit.",
            )
        )
        accepted = False

    if "\ufffd" in text:
        issues.append(
            ValidationIssue(
                "replacement_character",
                "warning",
                "Replacement character found; possible encoding damage.",
            )
        )

    if "\x00" in text:
        issues.append(
            ValidationIssue(
                "null_character",
                "warning",
                "Null byte found; possible extraction or encoding damage.",
            )
        )

    control_chars = [
        char for char in text
        if ord(char) < 32 and char not in "\n\r\t"
    ]
    if control_chars:
        issues.append(
            ValidationIssue(
                "control_characters",
                "warning",
                f"Found {len(control_chars)} unexpected control character(s).",
            )
        )

    if re.search(r"([^\w\s])\1{7,}", text):
        issues.append(
            ValidationIssue(
                "repeated_symbol_sequence",
                "warning",
                "Long repeated symbol sequence found; manual review recommended.",
            )
        )

    return ValidationResult(
        source_path=document.source_path,
        accepted=accepted,
        issues=tuple(issues),
    )
