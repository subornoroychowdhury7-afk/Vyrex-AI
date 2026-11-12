#!/usr/bin/env python3
"""Read-only OANC written_1 metadata distribution audit."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from vyrex_ai.data.oanc_metadata import read_oanc_metadata_file

OANC_ROOT = (
    PROJECT / "data" / "raw" / "oanc" / "extracted"
    / "OANC-GrAF" / "data" / "written_1"
)
REPORT_DIR = PROJECT / "reports"


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def write_counter_csv(path: Path, counter: Counter, label: str) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([label, "document_count"])
        for value, count in counter.most_common():
            writer.writerow([value, count])


def main() -> int:
    if not OANC_ROOT.is_dir():
        print(f"ERROR: OANC directory not found: {OANC_ROOT}")
        return 2

    text_files = sorted(OANC_ROOT.rglob("*.txt"))

    author_counts = Counter()
    publisher_counts = Counter()
    source_title_counts = Counter()
    category_counts = Counter()
    document_rows = []

    metadata_found = 0
    metadata_missing = 0
    metadata_errors = 0
    documents_with_authors = 0
    documents_with_publisher = 0
    documents_with_source_title = 0
    documents_with_categories = 0
    documents_with_no_grouping_metadata = 0

    for text_path in text_files:
        anc_path = text_path.with_suffix(".anc")

        if not anc_path.is_file():
            metadata_missing += 1
            document_rows.append({
                "document": text_path.relative_to(OANC_ROOT).as_posix(),
                "metadata_status": "MISSING",
                "document_id": "",
                "title": "",
                "authors": "",
                "publisher": "",
                "source_title": "",
                "publication_date": "",
                "word_count": "",
                "language": "",
                "language_code": "",
                "category_codes": "",
            })
            continue

        metadata_found += 1
        result = read_oanc_metadata_file(anc_path, text_path)

        if result.metadata is None:
            metadata_errors += 1
            document_rows.append({
                "document": text_path.relative_to(OANC_ROOT).as_posix(),
                "metadata_status": f"ERROR: {result.error}",
                "document_id": "",
                "title": "",
                "authors": "",
                "publisher": "",
                "source_title": "",
                "publication_date": "",
                "word_count": "",
                "language": "",
                "language_code": "",
                "category_codes": "",
            })
            continue

        metadata = result.metadata
        authors = tuple(dict.fromkeys(
            normalized(author) for author in metadata.authors if author.strip()
        ))
        publisher = normalized(metadata.publisher) if metadata.publisher else ""
        source_title = (
            normalized(metadata.source_title) if metadata.source_title else ""
        )
        categories = tuple(dict.fromkeys(
            code.strip().upper()
            for code in metadata.category_codes if code.strip()
        ))

        if authors:
            documents_with_authors += 1
            author_counts.update(authors)

        if publisher:
            documents_with_publisher += 1
            publisher_counts[publisher] += 1

        if source_title:
            documents_with_source_title += 1
            source_title_counts[source_title] += 1

        if categories:
            documents_with_categories += 1
            category_counts.update(categories)

        if not authors and not publisher and not source_title:
            documents_with_no_grouping_metadata += 1

        document_rows.append({
            "document": text_path.relative_to(OANC_ROOT).as_posix(),
            "metadata_status": "OK",
            "document_id": metadata.document_id or "",
            "title": metadata.title or "",
            "authors": " | ".join(metadata.authors),
            "publisher": metadata.publisher or "",
            "source_title": metadata.source_title or "",
            "publication_date": metadata.publication_date or "",
            "word_count": metadata.word_count if metadata.word_count is not None else "",
            "language": metadata.language or "",
            "language_code": metadata.language_code or "",
            "category_codes": " ".join(categories),
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_directory": str(OANC_ROOT),
        "documents_found": len(text_files),
        "metadata_files_found": metadata_found,
        "metadata_files_missing": metadata_missing,
        "metadata_parse_errors": metadata_errors,
        "documents_with_authors": documents_with_authors,
        "documents_with_publisher": documents_with_publisher,
        "documents_with_source_title": documents_with_source_title,
        "documents_with_categories": documents_with_categories,
        "documents_without_author_publisher_or_source_title":
            documents_with_no_grouping_metadata,
        "unique_authors": len(author_counts),
        "unique_publishers": len(publisher_counts),
        "unique_source_titles": len(source_title_counts),
        "unique_category_codes": len(category_counts),
        "top_authors": author_counts.most_common(20),
        "top_publishers": publisher_counts.most_common(20),
        "top_source_titles": source_title_counts.most_common(20),
        "category_distribution": dict(category_counts.most_common()),
        "grouping_warning": (
            "Frequency reports are descriptive only. They do not establish "
            "leakage-safe groups or assign dataset splits."
        ),
    }

    summary_path = REPORT_DIR / "oanc_written1_metadata_summary.json"
    documents_path = REPORT_DIR / "oanc_written1_document_metadata.csv"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with documents_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(document_rows[0].keys()) if document_rows else [
            "document", "metadata_status"
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(document_rows)

    write_counter_csv(
        REPORT_DIR / "oanc_written1_authors.csv",
        author_counts,
        "author",
    )
    write_counter_csv(
        REPORT_DIR / "oanc_written1_publishers.csv",
        publisher_counts,
        "publisher",
    )
    write_counter_csv(
        REPORT_DIR / "oanc_written1_source_titles.csv",
        source_title_counts,
        "source_title",
    )
    write_counter_csv(
        REPORT_DIR / "oanc_written1_categories.csv",
        category_counts,
        "category_code",
    )

    print("=" * 64)
    print("VYREX AI - OANC WRITTEN_1 METADATA AUDIT")
    print("=" * 64)
    print(f"DOCUMENTS: {len(text_files)}")
    print(f"METADATA FOUND: {metadata_found}")
    print(f"METADATA MISSING: {metadata_missing}")
    print(f"METADATA PARSE ERRORS: {metadata_errors}")
    print(f"DOCUMENTS WITH AUTHORS: {documents_with_authors}")
    print(f"DOCUMENTS WITH PUBLISHER: {documents_with_publisher}")
    print(f"DOCUMENTS WITH SOURCE TITLE: {documents_with_source_title}")
    print(f"DOCUMENTS WITHOUT GROUPING METADATA: {documents_with_no_grouping_metadata}")
    print()
    print(f"UNIQUE AUTHORS: {len(author_counts)}")
    print(f"UNIQUE PUBLISHERS: {len(publisher_counts)}")
    print(f"UNIQUE SOURCE TITLES: {len(source_title_counts)}")
    print(f"UNIQUE CATEGORY CODES: {len(category_counts)}")
    print()
    print("TOP 10 AUTHORS:")
    for name, count in author_counts.most_common(10):
        print(f"  {count:5}  {name}")
    print()
    print("TOP 10 PUBLISHERS:")
    for name, count in publisher_counts.most_common(10):
        print(f"  {count:5}  {name}")
    print()
    print("CATEGORY DISTRIBUTION:")
    for code, count in category_counts.most_common():
        print(f"  {code:8} {count}")
    print()
    print("REPORTS:")
    print(f"  {summary_path}")
    print(f"  {documents_path}")
    print(f"  {REPORT_DIR / 'oanc_written1_authors.csv'}")
    print(f"  {REPORT_DIR / 'oanc_written1_publishers.csv'}")
    print(f"  {REPORT_DIR / 'oanc_written1_source_titles.csv'}")
    print(f"  {REPORT_DIR / 'oanc_written1_categories.csv'}")
    print()
    print("READ-ONLY: no OANC source files were modified.")
    print("No training splits were assigned.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
