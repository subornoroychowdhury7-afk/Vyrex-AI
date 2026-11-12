from pathlib import Path

from vyrex_ai.data.oanc_metadata import (
    read_oanc_metadata,
    read_oanc_metadata_file,
)

CES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<cesHeader xmlns="http://www.xces.org/ns/GrAF/1.0/">
    <fileDesc>
        <titleStmt><title>TestDocument</title></titleStmt>
        <extent wordCount="296"/>
        <sourceDesc>
            <title>Test Corpus</title>
            <author>Author One</author>
            <author>Author Two</author>
            <publisher>Test Publisher</publisher>
            <pubDate value="2004">2004</pubDate>
        </sourceDesc>
    </fileDesc>
    <profileDesc>
        <langUsage>
            <language iso639="en-us">English (United States)</language>
        </langUsage>
        <textClass catRef="WR LT">
            <domain type="fundraising"/>
            <purpose type="education"/>
        </textClass>
        <primaryData loc="TestDocument.txt" medium="text"/>
    </profileDesc>
</cesHeader>
"""


def test_reads_normalized_ces_metadata(tmp_path: Path):
    metadata_file = tmp_path / "TestDocument.anc"
    metadata_file.write_text(CES_XML, encoding="utf-8")

    result = read_oanc_metadata_file(metadata_file)

    assert result.error is None
    assert result.metadata is not None
    assert result.metadata.document_id == "TestDocument"
    assert result.metadata.source_title == "Test Corpus"
    assert result.metadata.authors == ("Author One", "Author Two")
    assert result.metadata.publisher == "Test Publisher"
    assert result.metadata.publication_date == "2004"
    assert result.metadata.word_count == 296
    assert result.metadata.language_code == "en-us"
    assert result.metadata.category_codes == ("WR", "LT")


def test_preserves_collection_specific_xml_fields(tmp_path: Path):
    metadata_file = tmp_path / "TestDocument.anc"
    metadata_file.write_text(CES_XML, encoding="utf-8")

    result = read_oanc_metadata_file(metadata_file)

    assert result.metadata is not None
    raw_xml = result.metadata.raw_xml
    assert raw_xml["tag"] == "cesHeader"

    profile = raw_xml["children"][1]
    assert profile["tag"] == "profileDesc"
    assert any(
        child["tag"] == "textClass"
        for child in profile["children"]
    )


def test_reports_malformed_xml(tmp_path: Path):
    metadata_file = tmp_path / "broken.anc"
    metadata_file.write_text("<cesHeader>", encoding="utf-8")

    result = read_oanc_metadata_file(metadata_file)

    assert result.metadata is None
    assert result.error is not None
    assert result.error.startswith("ParseError:")


def test_missing_word_count_is_preserved_as_none(tmp_path: Path):
    xml = CES_XML.replace('<extent wordCount="296"/>', "<extent/>")
    metadata_file = tmp_path / "TestDocument.anc"
    metadata_file.write_text(xml, encoding="utf-8")

    result = read_oanc_metadata_file(metadata_file)

    assert result.metadata is not None
    assert result.metadata.word_count is None


def test_missing_metadata_file_is_reported(tmp_path: Path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Example text.", encoding="utf-8")

    result = read_oanc_metadata(text_file)

    assert result.metadata is None
    assert result.error is not None
    assert "Metadata file not found" in result.error


def test_reads_matching_metadata_file(tmp_path: Path):
    text_file = tmp_path / "document.txt"
    metadata_file = tmp_path / "document.anc"

    text_file.write_text("Example text.", encoding="utf-8")
    metadata_file.write_text(CES_XML, encoding="utf-8")

    result = read_oanc_metadata(text_file)

    assert result.error is None
    assert result.metadata is not None
    assert Path(result.metadata.document_path) == text_file.resolve()
