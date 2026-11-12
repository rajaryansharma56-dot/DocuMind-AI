
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

    if pdf_file.stat().st_size == 0:
        raise ValueError(
            "The PDF file is empty."
        )

    return pdf_file


# ==========================================
# 2. MAIN PDF PROCESSING FUNCTION
# ==========================================

def process_pdf(pdf_path):

    pdf_file = validate_pdf(pdf_path)

    try:

        with pymupdf.open(str(pdf_file)) as document:

            # Check password protection

            if document.needs_pass:
                raise ValueError(
                    "This PDF is password protected."
                )

            # Extract metadata

            metadata = document.metadata or {}

            # Extract text from every page

            extracted_pages = []

            empty_pages = []
            text_pages = 0
            total_characters = 0

            for page_number, page in enumerate(
                document, start=1
            ):

                text = page.get_text("text").strip()

                character_count = len(text)

                extracted_pages.append({

                    "page_number": page_number,

                    "text": text,

                    "character_count": character_count

                })

                total_characters += character_count

                if text:
                    text_pages += 1

                else:
                    empty_pages.append(page_number)

            # Document statistics

            total_pages = document.page_count

            file_size_kb = round(
                pdf_file.stat().st_size / 1024, 2
            )

            # Extract complete document information

            pdf_information = {

                "filename": pdf_file.name,

                "file_size_kb": file_size_kb,

                "total_pages": total_pages,

                "text_pages": text_pages,

                "empty_pages": len(empty_pages),

                "empty_page_numbers": empty_pages,

                "total_characters": total_characters,

                "title": metadata.get("title"),

                "author": metadata.get("author"),

                "subject": metadata.get("subject"),

                "keywords": metadata.get("keywords"),

                "creator": metadata.get("creator"),

                "producer": metadata.get("producer"),

                "creation_date": metadata.get("creationDate"),

                "modification_date": metadata.get("modDate"),

                "format": metadata.get("format"),

                "pages": extracted_pages

            }

            # Processing warnings

            warnings = []

            if total_characters == 0:
                warnings.append(
                    "No extractable text found. "
                    "This PDF may require OCR."
                )

            elif empty_pages:
                warnings.append(
                    "Some pages contain no extractable text."
                )

            pdf_information["warnings"] = warnings

            return pdf_information

    except pymupdf.FileDataError as error:

        raise ValueError(
            "The PDF is corrupted or invalid."
        ) from error


# ==========================================
# 3. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(pdf_path):

    result = process_pdf(pdf_path)

    return [
        {
            "page_number": page["page_number"],
            "text": page["text"]
        }
        for page in result["pages"]
    ]


# ==========================================
# 4. EXTRACT PDF METADATA
# ==========================================

def get_pdf_metadata(pdf_path):

    result = process_pdf(pdf_path)

    return {

        "filename": result["filename"],

        "file_size_kb": result["file_size_kb"],

        "total_pages": result["total_pages"],

        "empty_pages": result["empty_pages"],

        "text_pages": result["text_pages"],

        "title": result["title"],

        "author": result["author"]

    }
