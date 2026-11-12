
from src.pdf_processor import process_pdf
from src.text_chunker import create_chunks
from src.embedding_engine import generate_embeddings, get_embedding_info


PDF_PATH = "uploads/sample.pdf"


def main():

    print("\n========== DOCUMIND AI ==========")

    # Step 1: Extract PDF text
    pdf_data = process_pdf(PDF_PATH)

    # Step 2: Create text chunks
    chunks = create_chunks(pdf_data["pages"])

    # Step 3: Extract text from chunks
    chunk_texts = [chunk["text"] for chunk in chunks]

    # Step 4: Generate embeddings
    embeddings = generate_embeddings(chunk_texts)

    # Step 5: Display embedding information
    info = get_embedding_info(embeddings)

    print("\nEMBEDDING INFORMATION")

    for key, value in info.items():
        print(f"{key}: {value}")

    print("\nFIRST EMBEDDING PREVIEW")
    print(embeddings[0][:10])

    print("\n========== EMBEDDING COMPLETED ==========")


if __name__ == "__main__":
    main()
