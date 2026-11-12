"""Data ingestion, quality, splitting, and export for VYREX AI."""

from .corpus import CorpusStats, inspect_training_corpus, iter_training_texts
from .deduplication import DeduplicationResult, deduplicate_documents
from .export import ExportResult, export_dataset
from .manifest import ManifestEntry, build_manifest, write_manifest
from .oanc_metadata import (
    OANCMetadata,
    OANCMetadataResult,
    read_oanc_metadata,
    read_oanc_metadata_file,
)
from .pipeline import PipelineReport, prepare_dataset
from .readers import Document, ReadFailure, ReadResult, read_directory, read_document
from .splitting import DatasetSplits, split_documents
from .validation import ValidationIssue, ValidationResult, validate_document

__all__ = [
    "CorpusStats",
    "DatasetSplits",
    "DeduplicationResult",
    "Document",
    "ExportResult",
    "ManifestEntry",
    "OANCMetadata",
    "OANCMetadataResult",
    "PipelineReport",
    "ReadFailure",
    "ReadResult",
    "ValidationIssue",
    "ValidationResult",
    "build_manifest",
    "deduplicate_documents",
    "export_dataset",
    "inspect_training_corpus",
    "iter_training_texts",
    "prepare_dataset",
    "read_directory",
    "read_document",
    "read_oanc_metadata",
    "read_oanc_metadata_file",
    "split_documents",
    "validate_document",
    "write_manifest",
]
