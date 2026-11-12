
from src.llm import OllamaLLM

print("\n========== DOCUMIND LLM TEST ==========\n")

llm = None

try:
    llm = OllamaLLM(model="qwen3:4b")

    print("\nOllama connection successful!")

    answer = llm.generate(
        prompt="Explain what a vector database is in 3 short points."
    )

    print("\n========== AI RESPONSE ==========\n")
    print(answer)

    print("\n========== GENERATION STATISTICS ==========\n")
    print(llm.last_generation_stats)

except Exception as error:
    print("\nLLM TEST FAILED:")
    print(error)

finally:
    if llm:
        llm.close()

print("\n========== TEST FINISHED ==========")
