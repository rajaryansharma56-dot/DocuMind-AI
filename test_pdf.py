
from src.pdf_processor import extract_text_from_pdf

pdf_path = "data/uploads/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))

for page in pages:
    print("\nPage:", page["page_number"])
    print(page["text"][:500])
