
from src.pdf_processor import process_pdf

from src.text_chunker import (
    create_chunks,
    get_chunk_statistics
)

import json


# 1. Process PDF

PDF_PATH = "uploads/sample.pdf"

result = process_pdf(PDF_PATH)

if result.get("success") is False:
    print("PDF processing failed:", result["error"])
    exit()

# 2. Create chunks

chunks = create_chunks(
    result["pages"],
    chunk_size=1000,
    overlap=150
)

# 3. Calculate statistics

statistics = get_chunk_statistics(chunks)

# 4. Display results

print("\n========== DOCUMIND AI ==========")

print("\nCHUNK STATISTICS")

print(json.dumps(statistics, indent=4))

print("\nEXTRACTED CHUNKS")

for chunk in chunks:

    print("\n--------------------------------")

    print("Chunk ID:", chunk["chunk_id"])

    print("Page:", chunk["page_number"])

    print("Characters:", chunk["character_count"])

    print("Text:", chunk["text"][:200])

print("\n========== CHUNKING COMPLETED ==========")
