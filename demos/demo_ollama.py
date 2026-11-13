
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def ask_ai(question):

    payload = {
        "model": "qwen3:4b",
        "messages": [
            {
                "role": "system",
                "content": "You are DocuMind AI, an intelligent document assistant. Give accurate, clear and helpful answers."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False,
        "think": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"]


if __name__ == "__main__":

    print("\n========== DOCUMIND AI ==========\n")

    question = input("Ask your AI: ")

    answer = ask_ai(question)

    print("\nAI RESPONSE:\n")
    print(answer)
