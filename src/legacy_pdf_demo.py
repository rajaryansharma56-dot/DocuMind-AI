
import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path: str):

    pdf_file = Path(pdf_path)

    # Step 1: Check whether the PDF exists
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    # Step 2: Check whether it is a file
    if not pdf_file.is_file():
        raise ValueError("The given path is not a file.")

    # Step 3: Check the file extension
    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_pages = []

    try:
        # Step 4: Open the PDF safely
        with pymupdf.open(str(pdf_file)) as document:

            # Step 5: Check password protection
            if document.needs_pass:
                raise ValueError("This PDF is password protected.")

            # Step 6: Extract text from every page
            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text")

                extracted_pages.append({
                    "page_number": page_number,
                    "text": text.strip()
                })

    except pymupdf.FileDataError as error:
        raise ValueError("The PDF is corrupted or invalid.") from error

    return extracted_pages
