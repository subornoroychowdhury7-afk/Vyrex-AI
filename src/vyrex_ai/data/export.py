"""Export prepared VYREX datasets without modifying source documents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .manifest import build_manifest, write_manifest
from .oanc_metadata import read_oanc_metadata_file
from .pipeline import PipelineReport
from .readers import Document


@dataclass(frozen=True)
class ExportResult:
    """Paths and record counts for an exported dataset."""

    output_directory: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    manifest_path: Path
    train_count: int
    validation_count: int
    test_count: int


def _write_jsonl(
    documents: tuple[Document, ...],
    output_path: Path,
) -> None:
    """Write document text records using an atomic file replacement."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for document in documents:
                handle.write(
                    json.dumps(
                        {"text": document.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _collect_oanc_metadata(
    report: PipelineReport,
    corpus_root: str | Path | None = None,
) -> dict[str, dict]:
    """Collect metadata only when a matching OANC .anc file exists."""
    source_paths = {
        result.source_path for result in report.validation_results
    }
    source_paths.update(
        failure.source_path for failure in report.read_failures
    )
    source_paths.update(
        document.source_path
        for document in report.quarantined_documents
    )

    metadata_by_path: dict[str, dict] = {}

    for source_path in sorted(source_paths):
        metadata_path = Path(source_path).with_suffix(".anc")

        if not metadata_path.is_file():
            continue

        result = read_oanc_metadata_file(
            metadata_path=metadata_path,
            document_path=source_path,
            corpus_root=corpus_root,
        )

        metadata_record: dict = {"error": result.error}

        if result.metadata is not None:
            normalized = asdict(result.metadata)
            normalized.pop("raw_xml", None)
            metadata_record["metadata"] = normalized

        metadata_by_path[source_path] = metadata_record

    return metadata_by_path


def export_dataset(
    report: PipelineReport,
    source_groups: dict[str, str],
    output_directory: str | Path,
    *,
    corpus_root: str | Path | None = None,
) -> ExportResult:
    """Export prepared splits and an audit manifest to a new directory."""
    destination = Path(output_directory)

    splits = {
        "train": report.splits.train,
        "validation": report.splits.validation,
        "test": report.splits.test,
    }

    seen_paths: set[str] = set()

    for split_name, documents in splits.items():
        for document in documents:
            if not document.text.strip():
                raise ValueError(
                    f"Empty document found in {split_name}: "
                    f"{document.source_path}"
                )

            if document.source_path in seen_paths:
                raise ValueError(
                    "A document appears in more than one split: "
                    f"{document.source_path}"
                )

            seen_paths.add(document.source_path)

    destination.mkdir(parents=True, exist_ok=True)

    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    test_path = destination / "test.jsonl"
    manifest_path = destination / "manifest.jsonl"

    _write_jsonl(splits["train"], train_path)
    _write_jsonl(splits["validation"], validation_path)
    _write_jsonl(splits["test"], test_path)

    metadata_by_path = _collect_oanc_metadata(
        report, corpus_root=corpus_root
    )
    manifest_entries = build_manifest(
        report,
        source_groups,
        metadata_by_path=metadata_by_path,
    )
    write_manifest(manifest_entries, manifest_path)

    return ExportResult(
        output_directory=destination,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        manifest_path=manifest_path,
        train_count=len(splits["train"]),
        validation_count=len(splits["validation"]),
        test_count=len(splits["test"]),
    )
