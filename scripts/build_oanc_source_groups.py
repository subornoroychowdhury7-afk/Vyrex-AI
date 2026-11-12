from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath

COLLECTION_GROUPS = {
    "letters/icic": "collection:icic",
    "journal/slate": "collection:slate_unattributed",
    "journal/verbatim": "collection:verbatim",
    "fiction/eggan": "collection:eggan_fiction",
}


def normalize_author(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    return " ".join(name.casefold().split())


class DisjointSet:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        self.add(first)
        self.add(second)
        root_a = self.find(first)
        root_b = self.find(second)

        if root_a != root_b:
            # Deterministic root selection.
            smaller, larger = sorted((root_a, root_b))
            self.parent[larger] = smaller


def make_author_group_id(members: list[str]) -> str:
    canonical = sorted(members)
    slug = re.sub(r"[^a-z0-9]+", "-", canonical[0]).strip("-")
    digest = hashlib.sha256(
        "\n".join(canonical).encode("utf-8")
    ).hexdigest()[:8]
    return f"author:{slug or 'unknown'}-{digest}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-conscious OANC source groups."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/oanc_written1_document_metadata.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/oanc_written1_source_groups.json"),
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Metadata CSV not found: {args.input}")

    with args.input.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"document", "authors"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error(f"CSV is missing columns: {sorted(missing)}")
        documents = list(reader)

    if not documents:
        parser.error("Metadata CSV contains no documents.")

    dsu = DisjointSet()
    document_authors: dict[str, list[str]] = {}
    seen_paths: set[str] = set()

    # First pass: collect author identities and connect co-authors.
    for row in documents:
        relative_path = (row.get("document") or "").replace("\\", "/").strip()

        if not relative_path or relative_path in seen_paths:
            parser.error(f"Missing or duplicate document path: {relative_path!r}")

        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts:
            parser.error(f"Unsafe document path: {relative_path}")

        seen_paths.add(relative_path)

        raw_authors = row.get("authors") or ""
        authors = sorted({
            normalize_author(part)
            for part in raw_authors.split("|")
            if normalize_author(part)
            and normalize_author(part) != "anonymous"
            and normalize_author(part) != "lingua franca magazine"
        })

        document_authors[relative_path] = authors

        for author in authors:
            dsu.add(author)

        for author in authors[1:]:
            dsu.union(authors[0], author)

    # Build deterministic author connected components.
    components: dict[str, list[str]] = {}
    for author in sorted(dsu.parent):
        root = dsu.find(author)
        components.setdefault(root, []).append(author)

    author_group_ids = {
        author: make_author_group_id(members)
        for members in components.values()
        for author in members
    }

    # Second pass: assign exactly one group to every document.
    source_groups: dict[str, str] = {}
    group_counts: Counter[str] = Counter()

    for row in documents:
        relative_path = row["document"].replace("\\", "/").strip()
        authors = document_authors[relative_path]

        raw_author_names = {
            normalize_author(part)
            for part in (row.get("authors") or "").split("|")
            if normalize_author(part)
        }

        if "lingua franca magazine" in raw_author_names:
            group_id = "publication:lingua-franca-magazine"
        elif authors:
            group_id = author_group_ids[authors[0]]
        else:
            path_parts = PurePosixPath(relative_path).parts
            collection = "/".join(path_parts[:2])
            group_id = COLLECTION_GROUPS.get(collection)

            if group_id is None:
                parser.error(
                    f"Cannot safely group unattributed document: {relative_path}"
                )

        source_groups[relative_path] = group_id
        group_counts[group_id] += 1

    if len(source_groups) != len(documents):
        parser.error("Not every document received exactly one group.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")

    try:
        temporary_output.write_text(
            json.dumps(source_groups, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_output.replace(args.output)
    except OSError:
        temporary_output.unlink(missing_ok=True)
        raise

    print("VYREX AI - OANC SOURCE GROUP GENERATOR")
    print(f"DOCUMENTS: {len(documents)}")
    print(f"NAMED AUTHOR IDENTITIES: {len(dsu.parent)}")
    print(f"CONNECTED AUTHOR GROUPS: {len(components)}")
    print(f"TOTAL SOURCE GROUPS: {len(group_counts)}")
    print(f"ASSIGNED DOCUMENTS: {len(source_groups)}")
    print("\nCOLLECTION GROUPS:")

    for group_id in sorted(group_counts):
        if group_id.startswith("collection:"):
            print(f"  {group_id}: {group_counts[group_id]}")

    print(f"\nOUTPUT: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, csv.Error, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
