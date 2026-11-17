from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OANCMetadata:
    """Normalized CES and corpus metadata for one OANC document."""

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
    document_type: str | None = None
    collection: str | None = None
    genre: str | None = None


@dataclass(frozen=True)
class OANCMetadataResult:
    """Metadata extraction result, including any parsing failure."""

    metadata: OANCMetadata | None
    error: str | None = None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next(
        (child for child in element if _local_name(child.tag) == name),
        None,
    )


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


def _classify_document(
    document_path: str | Path,
    corpus_root: str | Path,
) -> tuple[str | None, str | None, str | None]:
    """Classify an OANC document using its corpus-relative path."""
    document = Path(document_path).resolve()
    root = Path(corpus_root).resolve()

    try:
        relative = document.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Document is outside OANC corpus root: {document}"
        ) from exc

    parts = relative.parts
    if len(parts) < 3 or parts[0].lower() != "data":
        return None, None, None

    section = parts[1].lower()
    subcategory = parts[2].lower()

    if section == "spoken":
        collection = parts[3].lower() if len(parts) > 4 else None
        return "spoken", collection, subcategory

    if section.startswith("written_"):
        return "written", section, subcategory

    return None, None, None


def read_oanc_metadata_file(
    metadata_path: str | Path,
    document_path: str | Path | None = None,
    corpus_root: str | Path | None = None,
) -> OANCMetadataResult:
    """Read and normalize one OANC .anc CES metadata header."""
    path = Path(metadata_path)

    try:
        raw_xml = path.read_bytes()
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            if "no element found" not in str(exc):
                raise

            repaired_xml = raw_xml.rstrip() + b"\\n</cesHeader>"
            root = ET.fromstring(repaired_xml)

            if (
                _local_name(root.tag) != "cesHeader"
                or not any(
                    _local_name(child.tag) == "cesHeader"
                    for child in root
                )
            ):
                raise exc

    except (OSError, ET.ParseError, UnicodeError) as exc:
        return OANCMetadataResult(
            metadata=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Some OANC headers contain a nested cesHeader.
    # Read metadata from the inner header when present.
    header = _first_child(root, "cesHeader") or root

    file_desc = _first_child(header, "fileDesc")
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

    document_type = collection = genre = None
    if document_path is not None and corpus_root is not None:
        try:
            document_type, collection, genre = _classify_document(
                document_path,
                corpus_root,
            )
        except ValueError as exc:
            return OANCMetadataResult(metadata=None, error=str(exc))

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
        document_type=document_type,
        collection=collection,
        genre=genre,
    )

    return OANCMetadataResult(metadata=metadata)


def read_oanc_metadata(
    text_path: str | Path,
    corpus_root: str | Path | None = None,
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
        corpus_root=corpus_root,
    )
