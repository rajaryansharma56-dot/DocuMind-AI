
import pymupdf
import os


def extract_pdf_metadata(pdf_path):

    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
        return

    document = pymupdf.open(pdf_path)

    metadata = document.metadata

    print("\n========== PDF METADATA ==========")

    print("File Name:", os.path.basename(pdf_path))
    print("Title:", metadata.get("title"))
    print("Author:", metadata.get("author"))
    print("Subject:", metadata.get("subject"))
    print("Creator:", metadata.get("creator"))
    print("Producer:", metadata.get("producer"))
    print("Number of Pages:", document.page_count)
    print("PDF Format:", metadata.get("format"))

    print("==================================")

    document.close()


extract_pdf_metadata("uploads/sample.pdf")
