
from src.retriever import Retriever


def main():

    print("\n========== DOCUMIND AI RETRIEVER ==========\n")

    retriever = Retriever()

    query = input("Ask a question about your PDF: ").strip()

    if not query:
        print("Please enter a valid question.")
        return

    print("\nSearching documents...\n")

    results = retriever.retrieve(query, top_k=3)

    if not results:
        print("No relevant information found.")
        return

    for i, result in enumerate(results, start=1):

        print(f"\nRESULT {i}")
        print("-" * 50)

        print("Page:", result["page_number"])
        print("Similarity:", result["similarity_score"])
        print("Content:", result["content"][:700])

    print("\n========== GENERATED CONTEXT ==========\n")

    print(retriever.get_context(query, top_k=3))


if __name__ == "__main__":
    main()
