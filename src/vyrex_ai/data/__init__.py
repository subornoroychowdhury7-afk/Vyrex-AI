"""Data ingestion, quality, splitting, and export for VYREX AI."""

from .corpus import CorpusStats, inspect_training_corpus, iter_training_texts
from .curriculum import (
    CURRICULUM_STAGES,
    CurriculumAssignment,
    cumulative_levels,
    load_curriculum_manifest,
    select_documents_for_stage,
    validate_assignment_coverage,
    validate_assignments,
    write_curriculum_manifest,
)
from .curriculum_export import (
    CurriculumExportResult,
    export_curriculum_datasets,
)
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
    "CURRICULUM_STAGES",
    "CorpusStats",
    "CurriculumAssignment",
    "CurriculumExportResult",
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
    "cumulative_levels",
    "deduplicate_documents",
    "export_curriculum_datasets",
    "export_dataset",
    "inspect_training_corpus",
    "iter_training_texts",
    "load_curriculum_manifest",
    "prepare_dataset",
    "read_directory",
    "read_document",
    "read_oanc_metadata",
    "read_oanc_metadata_file",
    "select_documents_for_stage",
    "split_documents",
    "validate_assignment_coverage",
    "validate_assignments",
    "validate_document",
    "write_curriculum_manifest",
    "write_manifest",
]
