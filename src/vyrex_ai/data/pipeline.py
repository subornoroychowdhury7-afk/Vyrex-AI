"""Non-destructive data preparation pipeline for VYREX AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .deduplication import DeduplicationResult, deduplicate_documents
from .readers import ReadFailure, read_directory
from .splitting import DatasetSplits, split_documents
from .validation import ValidationResult, validate_document


@dataclass(frozen=True)
class QuarantinedDocument:
    source_path: str
    reason: str


@dataclass(frozen=True)
class PipelineReport:
    total_read: int
    accepted_count: int
    rejected_count: int
    duplicate_count: int
    read_failures: tuple[ReadFailure, ...]
    validation_results: tuple[ValidationResult, ...]
    deduplication: DeduplicationResult
    splits: DatasetSplits
    quarantined_documents: tuple[QuarantinedDocument, ...] = ()


def prepare_dataset(
    input_directory: str | Path,
    source_groups: dict[str, str],
    output_ratios: tuple[float, float, float] = (0.98, 0.01, 0.01),
    seed: int = 42,
    min_chars: int = 100,
    max_chars: int = 2_000_000,
    exclusions: dict[str, str] | None = None,
) -> PipelineReport:
    """Read, validate, deduplicate, and split without modifying source files.

    Exclusions map resolved source paths to quarantine reasons.
    """
    read_result = read_directory(input_directory)
    exclusions = exclusions or {}

    excluded_documents = []
    active_documents = []

    for document in read_result.documents:
        reason = exclusions.get(document.source_path)

        if reason is not None:
            excluded_documents.append(
                QuarantinedDocument(
                    source_path=document.source_path,
                    reason=reason,
                )
            )
        else:
            active_documents.append(document)

    validation_results = tuple(
        validate_document(document, min_chars=min_chars, max_chars=max_chars)
        for document in active_documents
    )

    accepted_paths = {
        result.source_path
        for result in validation_results
        if result.accepted
    }

    accepted_documents = [
        document
        for document in active_documents
        if document.source_path in accepted_paths
    ]

    deduplication = deduplicate_documents(accepted_documents)

    if deduplication.unique_documents:
        splits = split_documents(
            deduplication.unique_documents,
            source_groups=source_groups,
            train_ratio=output_ratios[0],
            validation_ratio=output_ratios[1],
            test_ratio=output_ratios[2],
            seed=seed,
        )
    else:
        splits = DatasetSplits(train=(), validation=(), test=())

    return PipelineReport(
        total_read=len(read_result.documents),
        accepted_count=len(accepted_documents),
        rejected_count=len(active_documents) - len(accepted_documents),
        duplicate_count=len(deduplication.duplicates),
        read_failures=read_result.failures,
        validation_results=validation_results,
        deduplication=deduplication,
        splits=splits,
        quarantined_documents=tuple(excluded_documents),
    )
