"""Human-reviewed curriculum assignments for individual Tatoeba sentences."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


TATOEBA_LEVELS = (
    "basic_english",
    "elementary_intermediate_english",
    "upper_intermediate_english",
    "advanced_english",
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TatoebaAssignment:
    sentence_id: str
    level: str
    rationale: str
    reviewer: str
    reviewed_at: str
    source_sha256: str
    license: str


def _validate_assignment(item: TatoebaAssignment) -> TatoebaAssignment:
    if not isinstance(item, TatoebaAssignment):
        raise TypeError("Expected TatoebaAssignment.")

    values = (
        item.sentence_id,
        item.rationale,
        item.reviewer,
        item.reviewed_at,
        item.source_sha256,
        item.license,
    )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("All assignment fields must be non-empty strings.")

    if item.level not in TATOEBA_LEVELS:
        raise ValueError(f"Unsupported Tatoeba curriculum level: {item.level}")

    try:
        timestamp = datetime.fromisoformat(item.reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp.") from exc

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone.")

    if len(item.source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character SHA256 digest.")

    try:
        int(item.source_sha256, 16)
    except ValueError as exc:
        raise ValueError("source_sha256 must be hexadecimal.") from exc

    return TatoebaAssignment(
        sentence_id=item.sentence_id.strip(),
        level=item.level,
        rationale=item.rationale.strip(),
        reviewer=item.reviewer.strip(),
        reviewed_at=item.reviewed_at.strip(),
        source_sha256=item.source_sha256.lower(),
        license=item.license.strip(),
    )


def validate_tatoeba_assignments(
    assignments: Iterable[TatoebaAssignment],
) -> dict[str, TatoebaAssignment]:
    """Validate reviewed assignments and index them by sentence ID."""
    result = {}

    for raw in assignments:
        item = _validate_assignment(raw)

        if item.sentence_id in result:
            raise ValueError(f"Duplicate assignment: {item.sentence_id}")

        result[item.sentence_id] = item

    return result


def write_tatoeba_assignments(
    assignments: Iterable[TatoebaAssignment],
    output_path: str | Path,
) -> Path:
    """Write a validated, versioned assignment ledger atomically."""
    indexed = validate_tatoeba_assignments(assignments)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "assignments": [
            asdict(indexed[key]) for key in sorted(indexed)
        ],
    }

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return destination


def load_tatoeba_assignments(
    manifest_path: str | Path,
) -> dict[str, TatoebaAssignment]:
    """Load and validate a human-reviewed assignment ledger."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Tatoeba assignment manifest.")

    raw = payload.get("assignments")
    if not isinstance(raw, list):
        raise ValueError("'assignments' must be a JSON array.")

    assignments = []
    required = set(TatoebaAssignment.__dataclass_fields__)

    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"Invalid assignment fields at index {index}.")
        assignments.append(TatoebaAssignment(**item))

    return validate_tatoeba_assignments(assignments)


def build_tatoeba_review_batch(
    accepted_path: str | Path,
    output_directory: str | Path,
    *,
    offset: int = 0,
    limit: int = 5000,
) -> dict:
    """Create a bounded human-review batch without changing source records.

    Offset and limit apply to the accepted.jsonl record order. No level is
    inferred or pre-filled. Existing output directories are never overwritten.
    """
    source = Path(accepted_path)
    destination = Path(output_directory)

    if not source.is_file():
        raise FileNotFoundError(source)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer.")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    ))

    total = 0
    selected = 0

    try:
        queue_path = staging / "review_queue.jsonl"

        with source.open("r", encoding="utf-8") as input_handle, queue_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON at {source}:{line_number}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(f"Record at line {line_number} is not an object.")

                sentence_id = record.get("sentence_id")
                text = record.get("text")
                license_name = record.get("license")
                source_hash = record.get("source_sha256")

                if not all(isinstance(value, str) and value.strip() for value in (
                    sentence_id, text, license_name, source_hash
                )):
                    raise ValueError(
                        f"Missing sentence identity, text or provenance at line {line_number}."
                    )

                total += 1
                if total <= offset or selected >= limit:
                    continue

                words = text.split()
                queue_record = {
                    "sentence_id": sentence_id,
                    "text": text,
                    "license": license_name,
                    "author": record.get("author"),
                    "source_archive": record.get("source_archive"),
                    "source_sha256": source_hash,
                    "quality_bucket": record.get("quality_bucket"),
                    "quality_reasons": record.get("quality_reasons", []),
                    "signals": {
                        "token_count": len(words),
                        "character_count": len(text),
                        "average_token_length": (
                            round(sum(len(word) for word in words) / len(words), 2)
                            if words else 0
                        ),
                        "unique_token_ratio": (
                            round(len({word.casefold() for word in words}) / len(words), 3)
                            if words else 0
                        ),
                        "punctuation_count": sum(
                            not char.isalnum() and not char.isspace()
                            for char in text
                        ),
                    },
                    "review_status": "pending",
                    "assigned_level": None,
                }

                output_handle.write(
                    json.dumps(queue_record, ensure_ascii=False) + "\n"
                )
                selected += 1

            output_handle.flush()
            os.fsync(output_handle.fileno())

        template = {
            "schema_version": SCHEMA_VERSION,
            "assignments": [],
        }
        (staging / "assignments.template.json").write_text(
            json.dumps(template, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = {
            "schema_version": SCHEMA_VERSION,
            "source_file": str(source.resolve()),
            "source_records": total,
            "offset": offset,
            "requested_limit": limit,
            "selected_records": selected,
            "next_offset": offset + selected,
            "remaining_records": max(0, total - (offset + selected)),
            "levels": list(TATOEBA_LEVELS),
            "level_assignment": "human_review_only",
            "english_aptitude": "separate_original_content_required",
            "split_status": "not_split",
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

        os.replace(staging, destination)
        return summary

    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
