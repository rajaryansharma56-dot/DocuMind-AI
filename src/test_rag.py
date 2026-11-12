
from src.rag_pipeline import RAGPipeline

print("\n================================")
print("       DOCUMIND AI TEST")
print("================================\n")

rag = RAGPipeline()

while True:

    question = input("\nAsk a question (exit to quit): ")

    if question.lower() == "exit":
        print("DocuMind AI stopped.")
        break

    try:
        answer = rag.ask(question)

        print("\n========== DOCUMIND AI ANSWER ==========\n")
        print(answer)

    except Exception as e:
        print("\nError:", e)
