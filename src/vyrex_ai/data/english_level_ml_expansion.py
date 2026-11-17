"""Scalable provisional English-level candidate expansion."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from contextlib import closing
from pathlib import Path

from vyrex_ai.data.english_level_classifier_v2 import analyze_sentence
from vyrex_ai.data.english_level_ml import (
    FEATURE_NAMES,
    features_to_vector,
    validate_features,
)
from vyrex_ai.data.english_level_scoring_v2 import classify_features


SUPPORTED_LICENSES = {"CC0-1.0", "CC-BY-2.0-FR"}
DEFAULT_MAXIMUM_RECORDS = 10_000


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection without changing source text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _candidate_id(source_hash: str, sentence_id: str) -> str:
    payload = f"{source_hash.lower()}:{sentence_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _review_priority(prediction: dict) -> str:
    if prediction["review_required"]:
        return "high"

    score = prediction["complexity_score"]
    if any(abs(score - boundary) <= 10 for boundary in (20, 40, 65)):
        return "medium"

    return "normal"


def _temporary_file(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def expand_tatoeba_candidates(
    input_path: str | Path,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    maximum_records: int | None = DEFAULT_MAXIMUM_RECORDS,
    *,
    dry_run: bool = False,
) -> dict:
    """Stream accepted original Tatoeba sentences into a review queue.

    maximum_records limits nonblank input records examined.
    None explicitly permits processing the complete input.
    """

    source = Path(input_path).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")

    destination = Path(output_path).resolve() if output_path is not None else None
    manifest = Path(manifest_path).resolve() if manifest_path is not None else None

    if not dry_run and (destination is None or manifest is None):
        raise ValueError("output_path and manifest_path are required outside dry-run")

    paths = [source]
    paths.extend(path for path in (destination, manifest) if path is not None)

    if len(set(paths)) != len(paths):
        raise ValueError("Source, candidate, and manifest paths must differ")

    if not dry_run and (destination.exists() or manifest.exists()):
        raise FileExistsError("Output exists; refusing to overwrite")

    if maximum_records is not None and (
        isinstance(maximum_records, bool)
        or not isinstance(maximum_records, int)
        or maximum_records < 1
    ):
        raise ValueError("maximum_records must be a positive integer or None")

    priority_counts = Counter()
    level_counts = Counter()
    exclusion_counts = Counter()
    records_examined = 0
    candidates_created = 0
    license_counts = Counter()

    candidate_temp = None
    manifest_temp = None
    published_candidate = False

    try:
        if not dry_run:
            candidate_temp = _temporary_file(destination)
            manifest_temp = _temporary_file(manifest)

        with tempfile.TemporaryDirectory(prefix="vyrex-expansion-") as work:
            database = ":memory:" if dry_run else Path(work) / "dedup.sqlite3"

            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE sentence_ids (sentence_id TEXT PRIMARY KEY)"
                )
                connection.execute(
                    "CREATE TABLE normalized_texts (text TEXT PRIMARY KEY)"
                )

                with source.open("r", encoding="utf-8-sig") as input_handle:
                    with (
                        io.StringIO()
                        if dry_run
                        else candidate_temp.open(
                            "w", encoding="utf-8", newline="\n"
                        )
                    ) as output_handle:

                        for line_number, line in enumerate(input_handle, 1):
                            if not line.strip():
                                continue

                            if (
                                maximum_records is not None
                                and records_examined >= maximum_records
                            ):
                                break

                            records_examined += 1

                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ValueError(
                                    f"Malformed JSON at line {line_number}"
                                ) from exc

                            if not isinstance(record, dict):
                                raise ValueError(
                                    f"Record at line {line_number} "
                                    "must be an object"
                                )

                            sentence_id = record.get("sentence_id")
                            text = record.get("text")
                            license_name = record.get("license")

                            if (
                                not isinstance(sentence_id, str)
                                or not sentence_id.strip()
                            ):
                                raise ValueError(
                                    f"Invalid sentence ID at line {line_number}"
                                )

                            try:
                                connection.execute(
                                    "INSERT INTO sentence_ids VALUES (?)",
                                    (sentence_id,),
                                )
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(
                                    f"Duplicate sentence ID: {sentence_id}"
                                ) from exc

                            if not isinstance(text, str) or not text.strip():
                                exclusion_counts["empty_text"] += 1
                                continue

                            if record.get("language") != "eng":
                                exclusion_counts["not_english"] += 1
                                continue

                            if record.get("quality_bucket") != "accepted":
                                exclusion_counts["not_quality_accepted"] += 1
                                continue

                            if str(record.get("base_sentence_id")) != "0":
                                exclusion_counts["not_verified_original"] += 1
                                continue

                            if license_name not in SUPPORTED_LICENSES:
                                exclusion_counts["unsupported_license"] += 1
                                continue

                            source_hash = record.get("source_sha256")

                            if (
                                not isinstance(source_hash, str)
                                or len(source_hash) != 64
                            ):
                                raise ValueError(
                                    f"Invalid source hash at line {line_number}"
                                )

                            try:
                                int(source_hash, 16)
                            except ValueError as exc:
                                raise ValueError(
                                    f"Invalid source hash at line {line_number}"
                                ) from exc

                            source_hash = source_hash.lower()

                            if license_name == "CC-BY-2.0-FR":
                                attribution = record.get("attribution")

                                if (
                                    not isinstance(attribution, dict)
                                    or not isinstance(
                                        attribution.get("author"), str
                                    )
                                    or not attribution["author"].strip()
                                    or attribution.get("source") != "Tatoeba"
                                ):
                                    raise ValueError(
                                        "Missing CC-BY attribution "
                                        f"at line {line_number}"
                                    )

                            normalized = normalize_text(text)

                            try:
                                connection.execute(
                                    "INSERT INTO normalized_texts VALUES (?)",
                                    (normalized,),
                                )
                            except sqlite3.IntegrityError:
                                exclusion_counts[
                                    "duplicate_normalized_text"
                                ] += 1
                                continue

                            features = analyze_sentence(text)
                            validate_features(features)
                            vector = features_to_vector(features)
                            prediction = classify_features(features)

                            priority = _review_priority(prediction)

                            candidate = {
                                "candidate_id": _candidate_id(
                                    source_hash, sentence_id
                                ),
                                "sentence_id": sentence_id,
                                "text": text,
                                "language": "eng",
                                "license": license_name,
                                "author": record.get("author"),
                                "attribution": record.get("attribution"),
                                "source_archive": record.get("source_archive"),
                                "source_sha256": source_hash,
                                "base_sentence_id": record.get(
                                    "base_sentence_id"
                                ),
                                "quality_bucket": record["quality_bucket"],
                                "quality_reasons": record.get(
                                    "quality_reasons", []
                                ),
                                "features": features,
                                "feature_vector": vector,
                                "feature_names": list(FEATURE_NAMES),
                                "prediction": {
                                    "predicted_level": prediction[
                                        "predicted_level"
                                    ],
                                    "complexity_score": prediction[
                                        "complexity_score"
                                    ],
                                    "dimension_scores": prediction[
                                        "dimension_scores"
                                    ],
                                    "reasons": prediction["reasons"],
                                    "review_reasons": prediction[
                                        "review_reasons"
                                    ],
                                    "confidence": None,
                                    "confidence_type": "not_calibrated",
                                    "classification_status": (
                                        "provisional_not_human_approved"
                                    ),
                                },
                                "review_priority": priority,
                                "review_status": "pending",
                                "assigned_level": None,
                                "rationale": None,
                                "reviewer": None,
                                "reviewed_at": None,
                                "feature_schema_version": 1,
                            }

                            output_handle.write(
                                json.dumps(candidate, ensure_ascii=False)
                                + "\n"
                            )

                            candidates_created += 1
                            priority_counts[priority] += 1
                            level_counts[prediction["predicted_level"]] += 1
                            license_counts[license_name] += 1

                        output_handle.flush()
                        if not dry_run:
                            os.fsync(output_handle.fileno())

        summary = {
            "schema_version": 2,
            "source": str(source),
            "candidate_file": str(destination) if destination else None,
            "manifest_file": str(manifest) if manifest else None,
            "dry_run": dry_run,
            "records_examined": records_examined,
            "candidates_created": candidates_created,
            "maximum_records": maximum_records,
            "excluded": dict(sorted(exclusion_counts.items())),
            "license_counts": dict(sorted(license_counts.items())),
            "review_priority_counts": dict(sorted(priority_counts.items())),
            "provisional_level_counts": dict(sorted(level_counts.items())),
            "human_approved_records": 0,
            "confidence_calibrated": False,
            "production_model_approved": False,
            "source_modified": False,
        }

        if dry_run:
            return summary

        with manifest_temp.open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        if destination.exists() or manifest.exists():
            raise FileExistsError("Output appeared during processing")

        os.replace(candidate_temp, destination)
        published_candidate = True

        try:
            os.replace(manifest_temp, manifest)
        except Exception:
            destination.unlink()
            published_candidate = False
            raise

        return summary

    finally:
        for temporary in (candidate_temp, manifest_temp):
            if temporary is not None and temporary.exists():
                temporary.unlink()

        if published_candidate and not manifest.exists():
            if destination.exists():
                destination.unlink()
