import pytest

from vyrex_ai.data.readers import Document
from vyrex_ai.data.splitting import split_documents


def make_documents():
    documents = [
        Document("book_a_1.txt", "Book A chapter 1."),
        Document("book_a_2.txt", "Book A chapter 2."),
        Document("book_b_1.txt", "Book B chapter 1."),
        Document("book_b_2.txt", "Book B chapter 2."),
        Document("book_c_1.txt", "Book C chapter 1."),
        Document("book_d_1.txt", "Book D chapter 1."),
        Document("book_e_1.txt", "Book E chapter 1."),
        Document("book_f_1.txt", "Book F chapter 1."),
    ]

    groups = {
        "book_a_1.txt": "book_a",
        "book_a_2.txt": "book_a",
        "book_b_1.txt": "book_b",
        "book_b_2.txt": "book_b",
        "book_c_1.txt": "book_c",
        "book_d_1.txt": "book_d",
        "book_e_1.txt": "book_e",
        "book_f_1.txt": "book_f",
    }

    return documents, groups


def test_no_source_group_leakage():
    documents, groups = make_documents()

    splits = split_documents(
        documents,
        groups,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
    )

    split_groups = [
        {groups[document.source_path] for document in split}
        for split in (splits.train, splits.validation, splits.test)
    ]

    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])


def test_all_documents_are_preserved():
    documents, groups = make_documents()

    splits = split_documents(
        documents,
        groups,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
    )

    combined = splits.train + splits.validation + splits.test

    assert len(combined) == len(documents)
    assert {doc.source_path for doc in combined} == {
        doc.source_path for doc in documents
    }


def test_same_seed_produces_same_split():
    documents, groups = make_documents()

    first = split_documents(documents, groups, seed=42)
    second = split_documents(documents, groups, seed=42)

    assert first == second


def test_missing_source_group_is_rejected():
    documents, _ = make_documents()

    with pytest.raises(ValueError, match="Missing source group"):
        split_documents(documents, {})


def test_invalid_ratios_are_rejected():
    documents, groups = make_documents()

    with pytest.raises(ValueError, match="sum to 1"):
        split_documents(
            documents,
            groups,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.2,
        )


def test_empty_collection_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        split_documents([], {})


def test_zero_validation_and_test_are_supported():
    documents, groups = make_documents()

    splits = split_documents(
        documents,
        groups,
        train_ratio=1.0,
        validation_ratio=0.0,
        test_ratio=0.0,
    )

    assert len(splits.train) == len(documents)
    assert len(splits.validation) == 0
    assert len(splits.test) == 0
