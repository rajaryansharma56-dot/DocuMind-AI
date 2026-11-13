
from pathlib import Path

from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import create_chunks
from src.embedding_engine import generate_embeddings, generate_embedding
from src.vector_store import VectorStore


def main():

    print("\n========== DOCUMIND AI ==========\n")

    # Initialize vector database
    vector_store = VectorStore()

    # =====================================
    # 1. EXTRACT PDF
    # =====================================

    pdf_path = Path("data/uploads/sample.pdf")

    if not pdf_path.exists():
        print("PDF file not found!")
        return

    print("[1] Extracting PDF text...\n")

    pages = extract_text_from_pdf(str(pdf_path))

    print("Pages extracted:", len(pages))

    # =====================================
    # 2. CREATE CHUNKS
    # =====================================

    print("\n[2] Creating text chunks...\n")

    chunks = create_chunks(pages)

    print("Chunks created:", len(chunks))

    if not chunks:
        print("No chunks generated!")
        return

    # =====================================
    # 3. GENERATE EMBEDDINGS
    # =====================================

    print("\n[3] Generating embeddings...\n")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    print("Embeddings generated:", len(embeddings))

    # =====================================
    # 4. STORE IN VECTOR DATABASE
    # =====================================

    print("\n[4] Storing data in ChromaDB...\n")

    document_id = vector_store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        filename=pdf_path.name
    )

    print(
        "\nTotal chunks in database:",
        vector_store.get_collection_count()
    )

    # =====================================
    # 5. LIST STORED DOCUMENTS
    # =====================================

    print("\n[5] Stored documents...\n")

    documents = vector_store.list_documents()

    if not documents:
        print("No documents found.")

    for document in documents:

        print("Filename:", document["filename"])

        print("Document ID:", document["document_id"])

        print("Chunks:", document["chunks"])

        print("-" * 50)

    # =====================================
    # 6. SEMANTIC SEARCH
    # =====================================

    print("\n[6] Testing semantic search...\n")

    query = input(
        "Enter your search question: "
    ).strip()

    if not query:
        print("Search question cannot be empty!")
        return

    query_embedding = generate_embedding(query)

    results = vector_store.search(
        query_embedding,
        n_results=3
    )

    print("\n========== SEARCH RESULTS ==========\n")

    if not results:
        print("No search results found.")
        return

    for i, result in enumerate(results, start=1):

        metadata = result.get("metadata") or {}

        filename = metadata.get(
            "filename",
            "Unknown document"
        )

        page_number = metadata.get(
            "page_number",
            "Unknown"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown"
        )

        distance = result.get("distance", 0)

        content = result.get(
            "content",
            "No content available"
        )

        print(f"RESULT {i}")
        print("-" * 50)

        print("Document:", filename)

        print("Page:", page_number)

        print("Chunk:", chunk_index)

        print("Distance:", round(distance, 4))

        print("\nContent:")

        print(content[:700])

        print("\n")

    # =====================================
    # 7. DATABASE SUMMARY
    # =====================================

    print("\n========== DATABASE SUMMARY ==========\n")

    print(
        "Total stored chunks:",
        vector_store.get_collection_count()
    )

    print(
        "Total indexed documents:",
        len(vector_store.list_documents())
    )

    print("\nDOCUMIND AI TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
