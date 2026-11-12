"""Flexible CES metadata extraction for the Open American National Corpus."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OANCMetadata:
    """Normalized metadata extracted from one OANC CES header."""

    metadata_path: str
    document_path: str | None
    document_id: str | None
    title: str | None
    source_title: str | None
    authors: tuple[str, ...]
    publisher: str | None
    publication_date: str | None
    word_count: int | None
    language: str | None
    language_code: str | None
    category_codes: tuple[str, ...]
    raw_xml: dict


@dataclass(frozen=True)
class OANCMetadataResult:
    """Metadata extraction result, including any parsing failure."""

    metadata: OANCMetadata | None
    error: str | None = None


def _local_name(tag: str) -> str:
    """Remove an XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]


def _first_child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None

    for child in element:
        if _local_name(child.tag) == name:
            return child

    return None


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []

    return [
        child for child in element
        if _local_name(child.tag) == name
    ]


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None

    value = element.text.strip()
    return value or None


def _all_descendants(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []

    return [
        child for child in element.iter()
        if _local_name(child.tag) == name
    ]


def _raw_element(element: ET.Element) -> dict:
    """Preserve an XML element's tag, attributes, text, and child order."""
    return {
        "tag": _local_name(element.tag),
        "attributes": dict(element.attrib),
        "text": _text(element),
        "children": [_raw_element(child) for child in element],
    }


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_oanc_metadata_file(
    metadata_path: str | Path,
    document_path: str | Path | None = None,
) -> OANCMetadataResult:
    """Read and normalize one OANC .anc CES metadata header."""
    path = Path(metadata_path)

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError, UnicodeError) as exc:
        return OANCMetadataResult(
            metadata=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    file_desc = _first_child(root, "fileDesc")
    title_stmt = _first_child(file_desc, "titleStmt")
    extent = _first_child(file_desc, "extent")
    source_desc = _first_child(file_desc, "sourceDesc")

    profile_desc = _first_child(root, "profileDesc")
    lang_usage = _first_child(profile_desc, "langUsage")
    language_element = _first_child(lang_usage, "language")
    text_class = _first_child(profile_desc, "textClass")

    title = _text(_first_child(title_stmt, "title"))
    source_title = _text(_first_child(source_desc, "title"))

    authors = tuple(
        value
        for element in _all_descendants(source_desc, "author")
        if (value := _text(element)) is not None
    )

    publisher = _text(_first_child(source_desc, "publisher"))

    publication_element = _first_child(source_desc, "pubDate")
    publication_date = (
        _text(publication_element)
        or (
            publication_element.attrib.get("value")
            if publication_element is not None
            else None
        )
    )

    word_count = _optional_int(
        extent.attrib.get("wordCount") if extent is not None else None
    )

    language = _text(language_element)
    language_code = (
        language_element.attrib.get("iso639")
        if language_element is not None
        else None
    )

    category_codes = tuple(
        (text_class.attrib.get("catRef") or "").split()
    ) if text_class is not None else ()

    metadata = OANCMetadata(
        metadata_path=str(path.resolve()),
        document_path=(
            str(Path(document_path).resolve())
            if document_path is not None
            else None
        ),
        document_id=title,
        title=title,
        source_title=source_title,
        authors=authors,
        publisher=publisher,
        publication_date=publication_date,
        word_count=word_count,
        language=language,
        language_code=language_code,
        category_codes=category_codes,
        raw_xml=_raw_element(root),
    )

    return OANCMetadataResult(metadata=metadata)


def read_oanc_metadata(
    text_path: str | Path,
) -> OANCMetadataResult:
    """Find and read the .anc header beside an OANC text document."""
    document = Path(text_path)
    metadata_path = document.with_suffix(".anc")

    if not metadata_path.is_file():
        return OANCMetadataResult(
            metadata=None,
            error=f"Metadata file not found: {metadata_path}",
        )

    return read_oanc_metadata_file(
        metadata_path=metadata_path,
        document_path=document,
    )
