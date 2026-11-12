
from src.rag_pipeline import RAGPipeline

print("\n========== DOCUMIND FINAL PIPELINE TEST ==========\n")

try:

    rag = RAGPipeline()

    question = input("Enter your question: ")

    result = rag.ask(question)

    print("\n========== FINAL RESULT ==========\n")

    print("Question:", result["question"])

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(source)

    print("\nRetrieved chunks:", result["retrieved_chunks"])

    print("Execution time:", result["execution_time"], "seconds")

    print("\n========== TEST PASSED ==========\n")

except Exception as error:

    print("\n========== TEST FAILED ==========")
    print(error)
