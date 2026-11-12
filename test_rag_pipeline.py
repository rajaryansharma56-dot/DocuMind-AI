
from src.rag_pipeline import RAGPipeline


def main():

    print("\n========================================")
    print("          DOCUMIND AI")
    print("       ADVANCED RAG SYSTEM")
    print("========================================\n")

    pipeline = RAGPipeline(
        model="qwen3:4b"
    )

    print("Your PDF knowledge base is ready.")

    while True:

        question = input(
            "\nAsk a question (type exit to quit): "
        ).strip()

        if question.lower() in ["exit", "quit"]:

            print("\nThank you for using DocuMind AI!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        try:

            result = pipeline.ask(
                question=question,
                top_k=3
            )

            print("\n")
            print("=" * 55)
            print("             DOCUMIND AI ANSWER")
            print("=" * 55)

            print("\nQUESTION:")
            print(result["question"])

            print("\nANSWER:\n")
            print(result["answer"])

            print("\nSOURCES:")

            if result["sources"]:

                for source in result["sources"]:

                    print(
                        f"- {source['filename']} "
                        f"(Page {source['page_number']})"
                    )

            else:
                print("No sources available.")

            print(
                "\nRetrieved chunks:",
                result["retrieved_chunks"]
            )

            print("\n" + "=" * 55)

        except Exception as error:

            print("\nERROR:", error)


if __name__ == "__main__":
    main()
