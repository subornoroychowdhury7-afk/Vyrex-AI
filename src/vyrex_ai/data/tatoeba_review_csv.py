"""CSV export/import for human-reviewed Tatoeba curriculum decisions."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from .tatoeba_curriculum import (
    TATOEBA_LEVELS,
    TatoebaAssignment,
    write_tatoeba_assignments,
)

CSV_COLUMNS = (
    "sentence_id",
    "text",
    "license",
    "author",
    "attribution_source",
    "source_archive",
    "source_sha256",
    "review_status",
    "assigned_level",
    "rationale",
    "reviewer",
    "reviewed_at",
)

REVIEW_STATUSES = {
    "pending",
    "approved",
    "needs_review",
    "rejected",
}

SUPPORTED_LICENSES = {"CC0-1.0", "CC-BY-2.0-FR"}


def _atomic_csv_write(destination: Path, rows) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=CSV_COLUMNS,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")

        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def export_review_queue_csv(
    queue_path: str | Path,
    csv_path: str | Path,
) -> dict:
    """Export a review_queue.jsonl file without modifying its source."""
    source = Path(queue_path)
    destination = Path(csv_path)

    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    rows = []
    seen_ids = set()

    import json

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed review queue JSON at line {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Invalid review queue record at line {line_number}.")

            sentence_id = record.get("sentence_id")
            if not isinstance(sentence_id, str) or not sentence_id.strip():
                raise ValueError(f"Missing sentence ID at line {line_number}.")
            if sentence_id in seen_ids:
                raise ValueError(f"Duplicate sentence ID: {sentence_id}")
            seen_ids.add(sentence_id)

            text = record.get("text")
            license_name = record.get("license")
            source_hash = record.get("source_sha256")

            if not all(isinstance(value, str) and value.strip() for value in (
                text, license_name, source_hash
            )):
                raise ValueError(f"Missing text or provenance at line {line_number}.")

            if license_name not in SUPPORTED_LICENSES:
                raise ValueError(f"Unsupported license at line {line_number}.")

            author = record.get("author") or ""
            if license_name == "CC-BY-2.0-FR" and not author.strip():
                raise ValueError(
                    f"CC-BY attribution author missing at line {line_number}."
                )

            rows.append({
                "sentence_id": sentence_id,
                "text": text,
                "license": license_name,
                "author": author,
                "attribution_source": (
                    "Tatoeba" if license_name == "CC-BY-2.0-FR" else ""
                ),
                "source_archive": record.get("source_archive") or "",
                "source_sha256": source_hash.lower(),
                "review_status": "pending",
                "assigned_level": "",
                "rationale": "",
                "reviewer": "",
                "reviewed_at": "",
            })

    _atomic_csv_write(destination, rows)

    return {
        "csv_path": str(destination.resolve()),
        "records_exported": len(rows),
        "source_queue": str(source.resolve()),
        "source_modified": False,
    }


def import_review_decisions_csv(
    csv_path: str | Path,
    assignment_path: str | Path,
) -> dict:
    """Import approved CSV decisions into a validated assignment ledger.

    Pending, needs_review and rejected rows are excluded. Every approved
    row must contain a valid level, rationale, reviewer and timestamp.
    """
    source = Path(csv_path)
    destination = Path(assignment_path)

    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    assignments = []
    status_counts = {status: 0 for status in REVIEW_STATUSES}
    seen_ids = set()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError("CSV columns do not match the required review schema.")

        for line_number, row in enumerate(reader, 2):
            sentence_id = (row.get("sentence_id") or "").strip()
            if not sentence_id:
                raise ValueError(f"Missing sentence ID at CSV line {line_number}.")
            if sentence_id in seen_ids:
                raise ValueError(f"Duplicate sentence ID: {sentence_id}")
            seen_ids.add(sentence_id)

            status = (row.get("review_status") or "").strip().lower()
            if status not in REVIEW_STATUSES:
                raise ValueError(
                    f"Invalid review_status at CSV line {line_number}: {status}"
                )
            status_counts[status] += 1

            license_name = (row.get("license") or "").strip()
            if license_name not in SUPPORTED_LICENSES:
                raise ValueError(f"Unsupported license at CSV line {line_number}.")

            source_hash = (row.get("source_sha256") or "").strip().lower()
            if len(source_hash) != 64:
                raise ValueError(f"Invalid source hash at CSV line {line_number}.")
            try:
                int(source_hash, 16)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid source hash at CSV line {line_number}."
                ) from exc

            if license_name == "CC-BY-2.0-FR":
                if not (row.get("author") or "").strip():
                    raise ValueError(
                        f"CC-BY author missing at CSV line {line_number}."
                    )
                if (row.get("attribution_source") or "").strip() != "Tatoeba":
                    raise ValueError(
                        f"CC-BY attribution missing at CSV line {line_number}."
                    )

            if status != "approved":
                continue

            level = (row.get("assigned_level") or "").strip()
            if level not in TATOEBA_LEVELS:
                raise ValueError(
                    f"Invalid assigned level at CSV line {line_number}."
                )

            assignments.append(TatoebaAssignment(
                sentence_id=sentence_id,
                level=level,
                rationale=(row.get("rationale") or "").strip(),
                reviewer=(row.get("reviewer") or "").strip(),
                reviewed_at=(row.get("reviewed_at") or "").strip(),
                source_sha256=source_hash,
                license=license_name,
            ))

    if not assignments:
        raise ValueError("No approved review decisions found; ledger not created.")

    write_tatoeba_assignments(assignments, destination)

    return {
        "assignment_manifest": str(destination.resolve()),
        "approved_assignments": len(assignments),
        "excluded_rows": sum(
            count for status, count in status_counts.items()
            if status != "approved"
        ),
        "review_status_counts": status_counts,
    }
