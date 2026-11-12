from vyrex_ai.data.deduplication import (
    deduplicate_documents,
    fingerprint_document,
)
from vyrex_ai.data.readers import Document


def test_identical_documents_are_deduplicated():
    documents = [
        Document("a.txt", "Hello world."),
        Document("b.txt", "Hello world."),
    ]

    result = deduplicate_documents(documents)

    assert len(result.unique_documents) == 1
    assert len(result.duplicates) == 1
    assert result.duplicates[0].duplicate_of == "a.txt"


def test_whitespace_and_case_differences_are_normalized():
    documents = [
        Document("a.txt", "Hello   WORLD.\n"),
        Document("b.txt", "hello world."),
    ]

    result = deduplicate_documents(documents)

    assert len(result.unique_documents) == 1
    assert len(result.duplicates) == 1


def test_different_documents_are_preserved():
    documents = [
        Document("a.txt", "The sky is blue."),
        Document("b.txt", "The grass is green."),
    ]

    result = deduplicate_documents(documents)

    assert len(result.unique_documents) == 2
    assert len(result.duplicates) == 0


def test_first_occurrence_is_preserved():
    first = Document("first.txt", "Repeated content.")
    second = Document("second.txt", "Repeated content.")

    result = deduplicate_documents([first, second])

    assert result.unique_documents[0] is first
    assert result.duplicates[0].source_path == "second.txt"


def test_fingerprinting_is_deterministic():
    document = Document("a.txt", "Stable fingerprint.")
    assert fingerprint_document(document) == fingerprint_document(document)


def test_original_text_is_not_modified():
    original = "  Original   TEXT.\n"
    document = Document("a.txt", original)

    deduplicate_documents([document])

    assert document.text == original
