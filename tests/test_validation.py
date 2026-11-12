from vyrex_ai.data.readers import Document
from vyrex_ai.data.validation import validate_document


def make_document(text: str) -> Document:
    return Document(source_path="sample.txt", text=text)


def test_normal_document_is_accepted():
    result = validate_document(make_document("This is a normal document. " * 10))
    assert result.accepted is True
    assert result.issues == ()


def test_empty_document_is_rejected():
    result = validate_document(make_document("   \n\t"))
    assert result.accepted is False
    assert result.issues[0].code == "empty_document"


def test_short_document_gets_warning():
    result = validate_document(make_document("Short text."), min_chars=100)
    assert result.accepted is True
    assert "very_short_document" in [issue.code for issue in result.issues]


def test_replacement_character_is_flagged():
    result = validate_document(make_document("Example " * 20 + "\ufffd"))
    assert result.accepted is True
    assert "replacement_character" in [issue.code for issue in result.issues]


def test_null_character_is_flagged():
    result = validate_document(make_document("Example " * 20 + "\x00"))
    assert "null_character" in [issue.code for issue in result.issues]


def test_long_document_is_rejected():
    result = validate_document(make_document("A" * 101), max_chars=100)
    assert result.accepted is False
    assert "document_too_long" in [issue.code for issue in result.issues]


def test_repeated_symbols_are_flagged():
    result = validate_document(make_document("Example " * 20 + "!!!!!!!!"))
    assert "repeated_symbol_sequence" in [issue.code for issue in result.issues]


def test_invalid_limits_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        validate_document(make_document("Example"), min_chars=200, max_chars=100)
