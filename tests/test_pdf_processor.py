
import pytest
import pymupdf

from src.pdf_processor import (
    validate_pdf,
    process_pdf,
    extract_text_from_pdf,
    get_pdf_metadata
)


@pytest.fixture
def sample_pdf(tmp_path):

    pdf_path = tmp_path / "sample.pdf"

    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "DocuMind AI PDF testing."
    )

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "Second page for testing."
    )

    document.save(str(pdf_path))
    document.close()

    return pdf_path


def test_validate_valid_pdf(sample_pdf):

    result = validate_pdf(sample_pdf)

    assert result.exists()
    assert result.suffix == ".pdf"


def test_missing_pdf():

    with pytest.raises(FileNotFoundError):

        validate_pdf("missing.pdf")


def test_invalid_extension(tmp_path):

    file_path = tmp_path / "document.txt"

    file_path.write_text("Hello")

    with pytest.raises(ValueError):

        validate_pdf(file_path)


def test_empty_pdf(tmp_path):

    file_path = tmp_path / "empty.pdf"

    file_path.touch()

    with pytest.raises(ValueError):

        validate_pdf(file_path)


def test_directory_instead_of_file(tmp_path):

    with pytest.raises(ValueError):

        validate_pdf(tmp_path)


def test_process_pdf(sample_pdf):

    result = process_pdf(sample_pdf)

    assert result["total_pages"] == 2
    assert result["text_pages"] == 2
    assert result["empty_pages"] == 0

    assert result["total_characters"] > 0

    assert len(result["pages"]) == 2


def test_extract_text(sample_pdf):

    pages = extract_text_from_pdf(sample_pdf)

    assert len(pages) == 2

    assert "DocuMind AI PDF testing." in pages[0]["text"]


def test_get_pdf_metadata(sample_pdf):

    metadata = get_pdf_metadata(sample_pdf)

    assert metadata["filename"] == "sample.pdf"
    assert metadata["total_pages"] == 2

    assert "file_size_kb" in metadata


def test_corrupted_pdf(tmp_path):

    file_path = tmp_path / "corrupted.pdf"

    file_path.write_text("This is not a valid PDF.")

    with pytest.raises(ValueError):

        process_pdf(file_path)


def test_empty_page_detection(tmp_path):

    pdf_path = tmp_path / "empty_page.pdf"

    document = pymupdf.open()

    document.new_page()

    page = document.new_page()

    page.insert_text((72, 72), "Visible text")

    document.save(str(pdf_path))
    document.close()

    result = process_pdf(pdf_path)

    assert result["total_pages"] == 2
    assert result["empty_pages"] == 1
    assert result["empty_page_numbers"] == [1]


def test_text_extraction_compatibility(sample_pdf):

    result = process_pdf(sample_pdf)

    pages = extract_text_from_pdf(sample_pdf)

    assert len(result["pages"]) == len(pages)

    assert result["pages"][0]["text"] == pages[0]["text"]
