
import pytest

from src.text_chunker import (
    validate_chunk_settings,
    split_text,
    create_chunks,
    get_chunk_statistics
)


# ==========================================
# 1. VALIDATION TESTS
# ==========================================

def test_valid_chunk_settings():

    assert validate_chunk_settings(1000, 150) is None


@pytest.mark.parametrize("chunk_size, overlap", [
    (0, 0),
    (-10, 0),
    (100, -1),
    (100, 100),
    (100, 150),
    (True, 0),
    (100, False),
    (100.5, 10),
])

def test_invalid_chunk_settings(chunk_size, overlap):

    with pytest.raises(ValueError):

        validate_chunk_settings(chunk_size, overlap)


# ==========================================
# 2. TEXT SPLITTING TESTS
# ==========================================

def test_split_empty_text():

    assert split_text("", 100, 10) == []


def test_split_whitespace():

    assert split_text("   \n  ", 100, 10) == []


def test_split_short_text():

    text = "DocuMind AI testing"

    chunks = split_text(text, 100, 10)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_long_text():

    text = "word " * 100

    chunks = split_text(text, 50, 10)

    assert len(chunks) > 1

    assert all(chunks)


def test_whitespace_normalization():

    text = "DocuMind   AI\nAdvanced   RAG"

    chunks = split_text(text, 100, 10)

    assert chunks == ["DocuMind AI Advanced RAG"]


# ==========================================
# 3. DOCUMENT CHUNKING TESTS
# ==========================================

def test_create_chunks():

    pages = [
        {
            "page_number": 1,
            "text": "DocuMind AI is an advanced RAG system."
        }
    ]

    chunks = create_chunks(pages, 100, 10)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1


def test_unique_chunk_ids():

    pages = [
        {
            "page_number": 1,
            "text": "word " * 100
        }
    ]

    chunks = create_chunks(pages, 50, 10)

    ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_chunk_indexes():

    pages = [
        {
            "page_number": 1,
            "text": "word " * 100
        }
    ]

    chunks = create_chunks(pages, 50, 10)

    for index, chunk in enumerate(chunks):

        assert chunk["chunk_index"] == index


def test_page_number_preservation():

    pages = [
        {"page_number": 1, "text": "First page content."},
        {"page_number": 2, "text": "Second page content."}
    ]

    chunks = create_chunks(pages, 100, 10)

    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


def test_empty_pages_are_skipped():

    pages = [
        {"page_number": 1, "text": ""},
        {"page_number": 2, "text": "Valid content."}
    ]

    chunks = create_chunks(pages)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 2


def test_invalid_pages_type():

    with pytest.raises(TypeError):

        create_chunks("invalid input")


# ==========================================
# 4. CHUNK STATISTICS TESTS
# ==========================================

def test_chunk_statistics():

    chunks = [
        {
            "text": "Hello",
            "page_number": 1
        },
        {
            "text": "World!",
            "page_number": 2
        }
    ]

    stats = get_chunk_statistics(chunks)

    assert stats["total_chunks"] == 2
    assert stats["total_characters"] == 11
    assert stats["average_chunk_size"] == 5.5
    assert stats["minimum_chunk_size"] == 5
    assert stats["maximum_chunk_size"] == 6
    assert stats["unique_pages"] == 2


def test_empty_chunk_statistics():

    stats = get_chunk_statistics([])

    assert stats["total_chunks"] == 0
    assert stats["total_characters"] == 0
    assert stats["average_chunk_size"] == 0
    assert stats["unique_pages"] == 0


def test_invalid_statistics_input():

    with pytest.raises(TypeError):

        get_chunk_statistics("invalid")
