
from src.retriever import Retriever


def main():

    print("\n========== DOCUMIND AI ==========")
    print("ADVANCED HYBRID RETRIEVAL TEST\n")

    retriever = Retriever()

    print("Database chunks:",
          retriever.vector_store.get_collection_count())

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a valid question.")
        return

    results = retriever.retrieve(
        query=query,
        top_k=3
    )

    print("\n========== FINAL RESULTS ==========\n")

    if not results:
        print("No relevant information found.")
        return

    for index, result in enumerate(results, start=1):

        print(f"\nRESULT {index}")
        print("-" * 50)

        print("Filename:", result["filename"])
        print("Page:", result["page_number"])
        print("Chunk:", result["chunk_index"])

        print("RRF Score:", result["rrf_score"])
        print("Reranking Score:", result["rerank_score"])

        print("\nCONTENT:\n")
        print(result["content"])

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
