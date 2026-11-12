
import re
import requests


class OllamaLLM:

    def __init__(
        self,
        model="qwen3:4b",
        base_url="http://localhost:11434",
        timeout=240
    ):

        self.model = model
        self.api_url = f"{base_url.rstrip('/')}/api/chat"
        self.timeout = timeout

        self.session = requests.Session()

        self.options = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
            "num_predict": 1024
        }

    def clean_response(self, answer):

        if not answer:
            return ""

        answer = answer.strip()

        # Remove exposed thinking sections
        answer = re.sub(
            r"<think>.*?</think>",
            "",
            answer,
            flags=re.DOTALL
        )

        # Remove any remaining closing thinking tags
        answer = answer.replace("</think>", "")

        return answer.strip()

    def generate(self, prompt, system_prompt=None):

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not system_prompt:

            system_prompt = """
You are DocuMind AI, an advanced AI document assistant.

RULES:

1. Answer clearly and accurately.
2. Never invent facts.
3. Use supplied document context as evidence.
4. Identify missing information.
5. Never reveal internal reasoning.
6. Avoid unnecessary repetition.
7. Use headings and bullet points when helpful.

Return only the final answer.
"""

        payload = {

            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            "stream": False,

            # Disable thinking for concise RAG answers
            "think": False,

            "options": self.options
        }

        try:

            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()

            if "message" not in result:
                raise RuntimeError(
                    "Invalid response received from Ollama."
                )

            answer = result["message"].get("content", "")

            answer = self.clean_response(answer)

            if not answer:
                raise RuntimeError(
                    "The AI returned an empty response."
                )

            return answer

        except requests.exceptions.Timeout as error:

            raise RuntimeError(
                "Ollama request timed out. Try again."
            ) from error

        except requests.exceptions.ConnectionError as error:

            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Make sure the Ollama server is running."
            ) from error

        except requests.exceptions.HTTPError as error:

            raise RuntimeError(
                f"Ollama API error: {error}"
            ) from error

        except requests.exceptions.RequestException as error:

            raise RuntimeError(
                f"Ollama request failed: {error}"
            ) from error

        except (ValueError, KeyError) as error:

            raise RuntimeError(
                f"Invalid Ollama response: {error}"
            ) from error


if __name__ == "__main__":

    ai = OllamaLLM()

    print("\n========== DOCUMIND AI ==========\n")

    print("Model:", ai.model)

    print("Connecting to Ollama...\n")

    try:

        answer = ai.generate(
            prompt="Introduce yourself as DocuMind AI in three sentences."
        )

        print("========== AI RESPONSE ==========\n")
        print(answer)

    except RuntimeError as error:

        print("ERROR:", error)
