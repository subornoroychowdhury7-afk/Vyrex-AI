"""Dataset audit manifest generation for VYREX AI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .deduplication import fingerprint_document
from .pipeline import PipelineReport


@dataclass(frozen=True)
class ManifestEntry:
    source_path: str
    source_group: str | None
    status: str
    split: str | None
    sha256: str | None
    issues: tuple[dict, ...]
    duplicate_of: str | None = None
    failure_reason: str | None = None
    metadata: dict | None = None
    metadata_error: str | None = None
    exclusion_reason: str | None = None
    license_evidence: dict | None = None


def build_manifest(
    report: PipelineReport,
    source_groups: dict[str, str],
    metadata_by_path: dict[str, dict] | None = None,
) -> tuple[ManifestEntry, ...]:
    """Build an audit record for every read, excluded, or failed document."""
    metadata_by_path = metadata_by_path or {}
    license_by_path = {
        os.path.normcase(str(Path(record.source_path).resolve())): asdict(record)
        for record in report.license_records
    }
    def license_for(source_path: str) -> dict | None:
        key = os.path.normcase(str(Path(source_path).resolve()))
        return license_by_path.get(key)

    validation_by_path = {
        result.source_path: result
        for result in report.validation_results
    }

    split_by_path: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for document in getattr(report.splits, split_name):
            split_by_path[document.source_path] = split_name

    duplicate_by_path = {
        duplicate.source_path: duplicate
        for duplicate in report.deduplication.duplicates
    }

    unique_by_path = {
        document.source_path: document
        for document in report.deduplication.unique_documents
    }

    entries: list[ManifestEntry] = []

    for source_path, validation in validation_by_path.items():
        group = source_groups.get(source_path)
        issues = tuple(asdict(issue) for issue in validation.issues)
        metadata_record = metadata_by_path.get(source_path, {})
        metadata = metadata_record.get("metadata")
        metadata_error = metadata_record.get("error")

        if not validation.accepted:
            entries.append(
                ManifestEntry(
                    source_path=source_path,
                    source_group=group,
                    status="rejected",
                    split=None,
                    sha256=None,
                    issues=issues,
                    metadata=metadata,
                    metadata_error=metadata_error,
                    license_evidence=license_for(source_path),
                )
            )
            continue

        if source_path in duplicate_by_path:
            duplicate = duplicate_by_path[source_path]
            entries.append(
                ManifestEntry(
                    source_path=source_path,
                    source_group=group,
                    status="duplicate",
                    split=None,
                    sha256=duplicate.fingerprint,
                    issues=issues,
                    duplicate_of=duplicate.duplicate_of,
                    metadata=metadata,
                    metadata_error=metadata_error,
                    license_evidence=license_for(source_path),
                )
            )
            continue

        document = unique_by_path[source_path]
        entries.append(
            ManifestEntry(
                source_path=source_path,
                source_group=group,
                status="accepted",
                split=split_by_path.get(source_path),
                sha256=fingerprint_document(document),
                issues=issues,
                metadata=metadata,
                metadata_error=metadata_error,
                license_evidence=license_for(source_path),
            )
        )

    for quarantined in report.quarantined_documents:
        source_path = quarantined.source_path
        metadata_record = metadata_by_path.get(source_path, {})

        entries.append(
            ManifestEntry(
                source_path=source_path,
                source_group=source_groups.get(source_path),
                status="quarantined",
                split=None,
                sha256=None,
                issues=(),
                metadata=metadata_record.get("metadata"),
                metadata_error=metadata_record.get("error"),
                exclusion_reason=quarantined.reason,
                license_evidence=license_for(source_path),
            )
        )

    for failure in report.read_failures:
        metadata_record = metadata_by_path.get(failure.source_path, {})
        entries.append(
            ManifestEntry(
                source_path=failure.source_path,
                source_group=source_groups.get(failure.source_path),
                status="read_error",
                split=None,
                sha256=None,
                issues=(),
                failure_reason=failure.reason,
                metadata=metadata_record.get("metadata"),
                metadata_error=metadata_record.get("error"),
                license_evidence=license_for(failure.source_path),
            )
        )

    return tuple(sorted(entries, key=lambda entry: entry.source_path.casefold()))


def write_manifest(
    entries: tuple[ManifestEntry, ...],
    output_path: str | Path,
) -> Path:
    """Write manifest records as JSON Lines using an atomic replacement."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(
                    json.dumps(asdict(entry), ensure_ascii=False) + "\n"
                )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return destination
