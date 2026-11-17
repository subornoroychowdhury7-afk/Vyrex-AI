"""Provenance-preserving Tatoeba English sentence importer."""

from __future__ import annotations

import bz2
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import TextIO

MISSING = r"\N"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path, expected_fields: int):
    with bz2.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 1):
            if len(row) != expected_fields:
                raise ValueError(
                    f"{path.name}:{line_number}: expected "
                    f"{expected_fields} fields, got {len(row)}"
                )
            yield row


def _write(handle: TextIO, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def import_tatoeba(
    input_directory: str | Path,
    output_directory: str | Path,
) -> dict:
    """Import original English sentences; translations remain quarantined."""
    source = Path(input_directory)
    destination = Path(output_directory)

    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    detailed = source / "eng_sentences_detailed.tsv.bz2"
    base = source / "eng_sentences_base.tsv.bz2"
    cc0 = source / "eng_sentences_CC0.tsv.bz2"

    for path in (detailed, base, cc0):
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes = {p.name: _sha256(p) for p in (detailed, base, cc0)}

    # Load sentence relationships and the explicit CC0 export.
    base_ids: dict[str, str] = {}
    for sid, base_id in _rows(base, 2):
        if sid in base_ids:
            raise ValueError(f"Duplicate base sentence ID: {sid}")
        base_ids[sid] = base_id

    cc0_records: dict[str, tuple[str, str, str]] = {}
    for sid, language, text, modified in _rows(cc0, 4):
        if sid in cc0_records:
            raise ValueError(f"Duplicate CC0 sentence ID: {sid}")
        cc0_records[sid] = (language, text, modified)

    # Join detailed author/date metadata only for CC0 IDs.
    cc0_details: dict[str, tuple[str, str, str, str, str]] = {}
    for sid, language, text, author, added, modified in _rows(detailed, 6):
        if sid in cc0_records:
            if sid in cc0_details:
                raise ValueError(f"Duplicate detailed sentence ID: {sid}")
            cc0_details[sid] = (language, text, author, added, modified)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ))

    counts = Counter()
    seen_detailed: set[str] = set()

    try:
        cc0_out = staging / "cc0_candidates.jsonl"
        by_out = staging / "cc_by_candidates.jsonl"
        quarantine_out = staging / "quarantine.jsonl"

        with (
            cc0_out.open("w", encoding="utf-8", newline="\n") as cc0_handle,
            by_out.open("w", encoding="utf-8", newline="\n") as by_handle,
            quarantine_out.open(
                "w", encoding="utf-8", newline="\n"
            ) as quarantine_handle,
        ):
            # Explicit CC0 records, enriched from the detailed export.
            for sid, (language, text, modified) in cc0_records.items():
                base_id = base_ids.get(sid)
                detail = cc0_details.get(sid)

                record = {
                    "sentence_id": sid,
                    "text": text,
                    "language": language,
                    "license": "CC0-1.0",
                    "base_sentence_id": base_id,
                    "modified": modified,
                    "source_archive": cc0.name,
                    "source_sha256": hashes[cc0.name],
                }

                reason = None
                if language != "eng":
                    reason = "CC0_not_english"
                elif base_id != "0":
                    reason = "CC0_not_verified_original"
                elif not text.strip():
                    reason = "CC0_empty_text"
                elif detail is not None and (
                    detail[0] != language or detail[1] != text
                ):
                    reason = "CC0_detailed_text_or_language_mismatch"

                if detail is None:
                    record["author"] = None
                    record["author_metadata_status"] = (
                        "missing_detailed_record"
                    )
                    counts["cc0_missing_detailed"] += 1
                else:
                    _, _, author, added, detail_modified = detail
                    record["author"] = None if author == MISSING else author
                    record["date_added"] = added
                    record["date_modified"] = detail_modified
                    record["author_metadata_status"] = (
                        "joined" if author != MISSING
                        else "author_missing_in_detailed"
                    )
                    record["metadata_source_archive"] = detailed.name
                    record["metadata_source_sha256"] = hashes[detailed.name]
                    counts["cc0_detailed_matches"] += 1
                    if author == MISSING:
                        counts["cc0_author_missing"] += 1

                if reason:
                    record["quarantine_reason"] = reason
                    _write(quarantine_handle, record)
                    counts[f"quarantine_{reason}"] += 1
                else:
                    _write(cc0_handle, record)
                    counts["cc0_candidates"] += 1

            # Non-CC0 detailed records: only attributed originals are candidates.
            for sid, language, text, author, added, modified in _rows(
                detailed, 6
            ):
                if sid in seen_detailed:
                    raise ValueError(f"Duplicate detailed sentence ID: {sid}")
                seen_detailed.add(sid)

                if sid in cc0_records:
                    continue

                base_id = base_ids.get(sid)
                record = {
                    "sentence_id": sid,
                    "text": text,
                    "language": language,
                    "author": None if author == MISSING else author,
                    "date_added": added,
                    "date_modified": modified,
                    "base_sentence_id": base_id,
                    "source_archive": detailed.name,
                    "source_sha256": hashes[detailed.name],
                }

                if language != "eng":
                    reason = "not_english"
                elif base_id is None:
                    reason = "missing_base_relationship"
                elif base_id == MISSING:
                    reason = "unknown_original_relationship"
                elif base_id != "0":
                    reason = "translation_license_review_required"
                elif not text.strip():
                    reason = "empty_text"
                elif author == MISSING:
                    reason = "missing_author_attribution"
                else:
                    reason = None

                if reason:
                    record["quarantine_reason"] = reason
                    _write(quarantine_handle, record)
                    counts[f"quarantine_{reason}"] += 1
                else:
                    record["license"] = "CC-BY-2.0-FR"
                    record["attribution"] = {
                        "author": author,
                        "source": "Tatoeba",
                    }
                    _write(by_handle, record)
                    counts["cc_by_candidates"] += 1

        counts["detailed_records"] = len(seen_detailed)
        counts["base_records"] = len(base_ids)
        counts["cc0_export_records"] = len(cc0_records)
        counts["detailed_missing_base"] = len(
            seen_detailed - set(base_ids)
        )
        counts["base_missing_detailed"] = len(
            set(base_ids) - seen_detailed
        )

        summary = {
            "source": "Tatoeba English sentence exports",
            "schema_version": 2,
            "source_sha256": hashes,
            "counts": dict(sorted(counts.items())),
            "classification": "No CEFR labels assigned; human review required.",
            "split_status": "not_split",
        }

        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Publish the complete directory in one rename; never overwrite.
        os.replace(staging, destination)
        return summary

    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
