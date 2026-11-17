"""Deterministic, provenance-preserving Tatoeba quality filter."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
REPEATED = re.compile(r"(.)\1{5,}", re.UNICODE)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TOKEN = re.compile(r"\S+")


def classify_record(record: dict) -> tuple[str, list[str], int]:
    """Return deterministic bucket, reasons and whitespace-token count."""
    text = record.get("text")

    if not isinstance(text, str) or not text.strip():
        return "rejected", ["empty_or_invalid_text"], 0

    tokens = TOKEN.findall(text)
    count = len(tokens)
    reasons: list[str] = []

    if CONTROL.search(text):
        return "rejected", ["control_characters"], count

    if EMAIL.search(text):
        return "rejected", ["contains_email"], count

    if URL.search(text):
        return "rejected", ["contains_url"], count

    if REPEATED.search(text):
        return "rejected", ["long_character_repetition"], count

    punctuation = sum(
        1 for char in text if not char.isalnum() and not char.isspace()
    )
    if punctuation >= 8 and punctuation / max(len(text), 1) > 0.20:
        return "rejected", ["excessive_punctuation"], count

    if count < 4:
        reasons.append("short_sentence_manual_review")
    elif count > 60:
        reasons.append("very_long_sentence_manual_review")
    elif count > 30:
        reasons.append("long_sentence_manual_review")

    if len(text) > 200:
        reasons.append("over_200_characters_manual_review")

    if "..." in text or "…" in text:
        reasons.append("ellipsis_manual_review")

    if not reasons:
        return "accepted", [], count

    return "review", sorted(set(reasons)), count


def filter_tatoeba(
    input_directory: str | Path,
    output_directory: str | Path,
) -> dict:
    """Filter V2 candidates without modifying input files."""
    source = Path(input_directory)
    destination = Path(output_directory)

    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    inputs = {
        "cc0": source / "cc0_candidates.jsonl",
        "cc_by": source / "cc_by_candidates.jsonl",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ))

    counts = Counter()
    reason_counts = Counter()
    source_counts = Counter()

    try:
        with (
            (staging / "accepted.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as accepted,
            (staging / "review.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as review,
            (staging / "rejected.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as rejected,
        ):
            outputs = {
                "accepted": accepted,
                "review": review,
                "rejected": rejected,
            }

            for source_name, path in inputs.items():
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        try:
                            record = json.loads(line)
                            if not isinstance(record, dict):
                                raise ValueError("JSON record is not an object")
                        except (json.JSONDecodeError, ValueError) as exc:
                            bad = {
                                "source_file": path.name,
                                "source_line": line_number,
                                "raw_record": line.rstrip("\r\n"),
                                "quality_bucket": "rejected",
                                "quality_reasons": ["malformed_json"],
                                "quality_error": str(exc),
                            }
                            outputs["rejected"].write(
                                json.dumps(bad, ensure_ascii=False) + "\n"
                            )
                            counts["rejected"] += 1
                            source_counts[f"{source_name}_rejected"] += 1
                            reason_counts["malformed_json"] += 1
                            continue

                        bucket, reasons, token_count = classify_record(record)
                        record["quality_bucket"] = bucket
                        record["quality_reasons"] = reasons
                        record["quality_token_count"] = token_count

                        outputs[bucket].write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        counts[bucket] += 1
                        source_counts[f"{source_name}_{bucket}"] += 1
                        for reason in reasons:
                            reason_counts[reason] += 1

        total = sum(counts.values())
        summary = {
            "source": "Tatoeba V2 candidate files",
            "schema_version": 1,
            "policy": {
                "accepted_tokens": "4-30 inclusive",
                "review_tokens": "1-3 and over 30",
                "very_long_review_threshold": "over 60",
                "review_character_threshold": "over 200",
                "reject_email": True,
                "reject_url": True,
                "reject_control_characters": True,
                "reject_long_character_repetition": True,
                "reject_excessive_punctuation": True,
                "cefr_assignment": "none",
                "human_content_review": "still required",
            },
            "counts": {
                "accepted": counts["accepted"],
                "review": counts["review"],
                "rejected": counts["rejected"],
                "total": total,
            },
            "by_source": dict(sorted(source_counts.items())),
            "by_reason": dict(sorted(reason_counts.items())),
            "split_status": "not_split",
        }

        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        os.replace(staging, destination)
        return summary

    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
