
from src.pdf_processor import (
    process_pdf,
    extract_text_from_pdf,
    get_pdf_metadata
)

import json


PDF_PATH = "uploads/sample.pdf"


# Test 1: Complete PDF processing

print("\n========== COMPLETE PDF PROCESSING ==========")

result = process_pdf(PDF_PATH)

print(json.dumps(result, indent=4, ensure_ascii=False))


# Test 2: Text extraction compatibility

print("\n========== TEXT EXTRACTION TEST ==========")

pages = extract_text_from_pdf(PDF_PATH)

print("Extracted pages:", len(pages))


# Test 3: Metadata compatibility

print("\n========== METADATA TEST ==========")

metadata = get_pdf_metadata(PDF_PATH)

print(json.dumps(metadata, indent=4))


print("\n========== ALL TESTS COMPLETED ==========")
