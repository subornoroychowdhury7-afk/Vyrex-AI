from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .readers import Document


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve()))


@dataclass(frozen=True, slots=True)
class LicenseRecord:
    source_path: str
    source_name: str
    domain: str
    license_id: str
    source_url: str
    license_url: str
    training_use: bool
    commercial_use: bool
    redistribution: bool
    attribution_required: bool
    share_alike: bool
    review_status: str
    verified_at: str
    reviewer: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> LicenseRecord:
        required = {
            "source_path", "source_name", "domain", "license_id",
            "source_url", "license_url", "training_use",
            "commercial_use", "redistribution", "attribution_required",
            "share_alike", "review_status", "verified_at", "reviewer",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Missing license fields: {sorted(missing)}")

        for field in (
            "training_use", "commercial_use", "redistribution",
            "attribution_required", "share_alike",
        ):
            if type(value[field]) is not bool:
                raise ValueError(f"{field} must be a boolean")

        record = cls(**{key: value[key] for key in required})

        for field in (
            "source_path", "source_name", "domain", "license_id",
            "source_url", "license_url", "review_status",
            "verified_at", "reviewer",
        ):
            if not isinstance(getattr(record, field), str):
                raise ValueError(f"{field} must be a string")

        return record

    def ineligibility_reason(self) -> str | None:
        if self.review_status.strip().lower() != "approved":
            return "License has not received approved review status"
        if not self.license_id.strip():
            return "Missing exact license identifier"
        if not self.source_name.strip() or not self.domain.strip():
            return "Missing source name or domain"
        if not self.source_url.startswith("https://"):
            return "Missing valid HTTPS source evidence URL"
        if not self.license_url.startswith("https://"):
            return "Missing valid HTTPS license evidence URL"
        if not self.verified_at.strip() or not self.reviewer.strip():
            return "Missing verification date or reviewer"
        if self.training_use is not True:
            return "Training use is not explicitly permitted"
        if self.commercial_use is not True:
            return "Commercial use is not explicitly permitted"
        return None


def load_license_registry(
    registry_path: str | Path,
) -> dict[str, LicenseRecord]:
    """Load JSONL license records; reject malformed or duplicate entries."""
    path = Path(registry_path)
    records: dict[str, LicenseRecord] = {}

    if not path.is_file():
        return records

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("Expected a JSON object")

                record = LicenseRecord.from_mapping(raw)
                key = _path_key(record.source_path)

                if key in records:
                    raise ValueError("Duplicate source path in registry")

                records[key] = record

            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid license registry line {line_number}: {exc}"
                ) from exc

    return records


def build_license_exclusions(
    documents: Iterable[Document],
    records: dict[str, LicenseRecord],
) -> dict[str, str]:
    """Exclude unregistered or ineligible documents by default."""
    exclusions: dict[str, str] = {}

    for document in documents:
        key = _path_key(document.source_path)
        record = records.get(key)

        if record is None:
            exclusions[document.source_path] = (
                "No verified license record; quarantined by default"
            )
            continue

        reason = record.ineligibility_reason()
        if reason is not None:
            exclusions[document.source_path] = reason

    return exclusions
