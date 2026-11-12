
import re


# ==========================================
# 1. VALIDATE CHUNK SETTINGS
# ==========================================

def validate_chunk_settings(chunk_size, overlap):

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("Chunk size must be an integer.")

    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError("Overlap must be an integer.")

    if chunk_size < 1:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )


# ==========================================
# 2. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text(text, chunk_size, overlap):

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Prefer breaking at a word boundary
        if end < text_length:

            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(0, end - overlap)

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ==========================================
# 3. CREATE DOCUMENT CHUNKS
# ==========================================

def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
):

    validate_chunk_settings(chunk_size, overlap)

    if not isinstance(pages, list):
        raise TypeError("Pages must be provided as a list.")

    all_chunks = []

    chunk_counter = 0

    for page in pages:

        if not isinstance(page, dict):
            continue

        text = page.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        page_number = page.get("page_number", 0)

        page_chunks = split_text(
            text,
            chunk_size,
            overlap
        )

        for chunk_text in page_chunks:

            all_chunks.append({

                "chunk_id": f"chunk_{chunk_counter:06d}",

                "chunk_index": chunk_counter,

                "page_number": page_number,

                "text": chunk_text,

                "content": chunk_text,

                "character_count": len(chunk_text)

            })

            chunk_counter += 1

    return all_chunks


# ==========================================
# 4. CHUNK STATISTICS
# ==========================================

def get_chunk_statistics(chunks):

    if not isinstance(chunks, list):
        raise TypeError("Chunks must be a list.")

    if not chunks:

        return {
            "total_chunks": 0,
            "total_characters": 0,
            "average_chunk_size": 0,
            "minimum_chunk_size": 0,
            "maximum_chunk_size": 0,
            "unique_pages": 0
        }

    sizes = [
        len(chunk["text"])
        for chunk in chunks
    ]

    pages = {
        chunk["page_number"]
        for chunk in chunks
    }

    return {

        "total_chunks": len(chunks),

        "total_characters": sum(sizes),

        "average_chunk_size": round(
            sum(sizes) / len(sizes), 2
        ),

        "minimum_chunk_size": min(sizes),

        "maximum_chunk_size": max(sizes),

        "unique_pages": len(pages)

    }
