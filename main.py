
from src.retriever import Retriever
from src.answer_generator import AnswerGenerator


class DocuMindAI:

    def __init__(self):

        print("\nInitializing DocuMind AI...\n")

        self.retriever = Retriever()
        self.answer_generator = AnswerGenerator()

        print("Hybrid Retriever initialized.")
        print("Answer Generator initialized.")
        print("DocuMind AI is ready.")

    # -----------------------------------------
    # ASK QUESTION
    # -----------------------------------------

    def ask_question(self, question):

        if not isinstance(question, str) or not question.strip():

            return "Please enter a valid question."

        try:

            print("\nSearching PDF knowledge base...\n")

            # Step 1: Hybrid retrieval + reranking
            results = self.retriever.retrieve(
                question,
                top_k=5
            )

            if not results:

                return (
                    "No relevant information found "
                    "in your documents."
                )

            print(
                f"Retrieved {len(results)} relevant chunks."
            )

            # Step 2: Generate answer
            print("\nGenerating answer with Qwen3:4B...\n")

            answer = self.answer_generator.generate_answer(
                question,
                results
            )

            return answer

        except Exception as error:

            print(f"\nPipeline Error: {error}")

            return (
                "An error occurred while processing "
                "your question. Check the terminal "
                "for details."
            )

    # -----------------------------------------
    # CLOSE RESOURCES
    # -----------------------------------------

    def close(self):

        if self.answer_generator:

            self.answer_generator.close()


# ---------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------

def main():

    print("""
========================================
             DOCUMIND AI
          ADVANCED RAG SYSTEM
========================================
""")

    app = None

    try:

        app = DocuMindAI()

        print("\nYour PDF knowledge base is ready.")

        while True:

            print("\n" + "=" * 50)

            question = input(
                "\nAsk a question (type 'exit' to quit): "
            ).strip()

            if question.lower() in ["exit", "quit"]:

                print("\nThank you for using DocuMind AI!")
                break

            if not question:
                print("Please enter a valid question.")
                continue

            answer = app.ask_question(question)

            print("\n========== DOCUMIND AI ANSWER ==========\n")

            print(answer)

            print("\n========================================")

    except KeyboardInterrupt:

        print("\n\nDocuMind AI stopped.")

    except Exception as error:

        print(f"\nInitialization failed: {error}")

    finally:

        if app is not None:

            app.close()

            print("\nResources closed successfully.")


if __name__ == "__main__":
    main()
