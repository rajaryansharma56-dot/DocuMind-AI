
from src.retriever import Retriever
from src.generator import AnswerGenerator


def main():

    print("\n========== DOCUMIND AI ==========\n")

    # Initialize components
    retriever = Retriever()
    generator = AnswerGenerator()

    # Get user question
    question = input(
        "Ask your question about your PDF: "
    ).strip()

    if not question:
        print("Please enter a valid question.")
        return

    # Retrieve relevant document chunks
    print("\nSearching your documents...\n")

    results = retriever.retrieve(
        query=question,
        top_k=3
    )

    if not results:
        print("No relevant document content found.")
        return

    print("Retrieved sources:")

    for index, result in enumerate(results, start=1):

        print(f"\nSource {index}")
        print("Document:", result["filename"])
        print("Page:", result["page_number"])
        print("Distance:", result["distance"])

    # Generate answer using Qwen3
    print("\nGenerating AI answer with Qwen3:4b...\n")

    try:

        answer = generator.generate_answer(
            question,
            results
        )

        print("\n========== DOCUMIND AI ANSWER ==========\n")
        print(answer)

    except Exception as error:

        print("\nAnswer generation failed:")
        print(error)


if __name__ == "__main__":
    main()
