
import pymupdf
from pathlib import Path


# ==========================================
# 1. VALIDATE PDF FILE
# ==========================================

def validate_pdf(pdf_path):

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_file}"
        )

    if not pdf_file.is_file():
        raise ValueError(
            "The given path is not a file."
        )

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    return pdf_file


# ==========================================
# 2. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(pdf_path):

    pdf_file = validate_pdf(pdf_path)

    extracted_pages = []

    try:

        with pymupdf.open(str(pdf_file)) as document:

            if document.needs_pass:
                raise ValueError(
                    "This PDF is password protected."
                )

            for page_number, page in enumerate(
                document, start=1
            ):

                text = page.get_text("text")

                extracted_pages.append({

                    "page_number": page_number,

                    "text": text.strip()

                })

    except pymupdf.FileDataError as error:

        raise ValueError(
            "The PDF is corrupted or invalid."
        ) from error

    return extracted_pages


# ==========================================
# 3. EXTRACT PDF METADATA
# ==========================================

def get_pdf_metadata(pdf_path):

    pdf_file = validate_pdf(pdf_path)

    try:

        with pymupdf.open(str(pdf_file)) as document:

            if document.needs_pass:
                raise ValueError(
                    "This PDF is password protected."
                )

            total_pages = len(document)

            empty_pages = 0
            text_pages = 0

            for page in document:

                text = page.get_text("text").strip()

                if text:
                    text_pages += 1

                else:
                    empty_pages += 1

            metadata = document.metadata or {}

            pdf_information = {

                "filename": pdf_file.name,

                "file_size_kb": round(
                    pdf_file.stat().st_size / 1024, 2
                ),

                "total_pages": total_pages,

                "empty_pages": empty_pages,

                "text_pages": text_pages,

                "title": metadata.get("title"),

                "author": metadata.get("author")

            }

            return pdf_information

    except pymupdf.FileDataError as error:

        raise ValueError(
            "The PDF is corrupted or invalid."
        ) from error
