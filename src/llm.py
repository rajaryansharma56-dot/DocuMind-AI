
import re
import time
import requests

from requests.adapters import HTTPAdapter


class OllamaLLM:

    def __init__(
        self,
        model="qwen3:4b",
        base_url="http://localhost:11434",
        timeout=(10, 300),
        options=None
    ):

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"
        self.timeout = timeout

        # Generation settings
        self.options = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 30,
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
            "num_predict": 1024
        }

        if options:
            self.options.update(options)

        # Persistent HTTP session
        self.session = requests.Session()

        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=0
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.last_generation_stats = {}

        self._check_connection()

    # ---------------------------------
    # CHECK OLLAMA CONNECTION
    # ---------------------------------

    def _check_connection(self):

        try:

            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )

            response.raise_for_status()

            data = response.json()

            available_models = [
                item.get("name", "")
                for item in data.get("models", [])
            ]

            if self.model not in available_models:

                raise RuntimeError(
                    f"Model '{self.model}' is not installed.\n"
                    f"Available models: {available_models}"
                )

            print(
                f"Ollama connected successfully: {self.model}"
            )

        except requests.RequestException as error:

            raise RuntimeError(
                f"Could not connect to Ollama: {error}"
            ) from error

        except ValueError as error:

            raise RuntimeError(
                f"Invalid Ollama response: {error}"
            ) from error

    # ---------------------------------
    # CLEAN MODEL RESPONSE
    # ---------------------------------

    @staticmethod
    def _clean_response(answer):

        if not isinstance(answer, str):
            return ""

        answer = re.sub(
            r"<think>.*?</think>",
            "",
            answer,
            flags=re.DOTALL | re.IGNORECASE
        )

        answer = re.sub(
            r"</?think>",
            "",
            answer,
            flags=re.IGNORECASE
        )

        return answer.strip()

    # ---------------------------------
    # GENERATE AI RESPONSE
    # ---------------------------------

    def generate(self, prompt, system_prompt=None):

        if not isinstance(prompt, str) or not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

        payload = {

            "model": self.model,

            "prompt": f"{prompt.strip()}\n/no_think",

            "system": system_prompt or "",

            "stream": False,

            "think": False,

            "keep_alive": "10m",

            "options": self.options.copy()
        }

        try:

            start_time = time.perf_counter()

            response = self.session.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()

            if not isinstance(result, dict):

                raise RuntimeError(
                    "Invalid Ollama response format."
                )

            raw_answer = result.get("response")

            if not isinstance(raw_answer, str):

                raise RuntimeError(
                    "Ollama response field is missing or invalid."
                )

            answer = self._clean_response(raw_answer)

            if not answer:

                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            elapsed = time.perf_counter() - start_time

            generated_tokens = result.get("eval_count", 0) or 0

            eval_duration = result.get("eval_duration", 0) or 0

            tokens_per_second = 0

            if eval_duration > 0:

                tokens_per_second = (
                    generated_tokens /
                    (eval_duration / 1_000_000_000)
                )

            # Check completion information
            done = result.get("done", False)

            done_reason = result.get(
                "done_reason",
                "unknown"
            )

            # Save generation statistics
            self.last_generation_stats = {

                "model": self.model,

                "generation_time": round(elapsed, 2),

                "prompt_tokens": result.get(
                    "prompt_eval_count", 0
                ),

                "generated_tokens": generated_tokens,

                "tokens_per_second": round(
                    tokens_per_second, 2
                ),

                "load_duration": result.get(
                    "load_duration", 0
                ),

                "total_duration": result.get(
                    "total_duration", 0
                ),

                "done": done,

                "done_reason": done_reason,

                "possibly_truncated": (
                    done_reason == "length"
                )
            }

            print(
                f"\nGeneration completed in {elapsed:.2f} seconds."
            )

            print(
                f"Generated tokens: {generated_tokens}"
            )

            print(
                f"Generation speed: {tokens_per_second:.2f} tokens/sec"
            )

            print(
                f"Completion reason: {done_reason}"
            )

            return answer

        except requests.Timeout as error:

            raise RuntimeError(
                "Ollama timed out. Reduce context or retry."
            ) from error

        except requests.HTTPError as error:

            details = (
                error.response.text[:500]
                if error.response is not None
                else str(error)
            )

            raise RuntimeError(
                f"Ollama HTTP error: {details}"
            ) from error

        except requests.RequestException as error:

            raise RuntimeError(
                f"Ollama communication failed: {error}"
            ) from error

        except ValueError as error:

            raise RuntimeError(
                f"Invalid JSON response from Ollama: {error}"
            ) from error

    # ---------------------------------
    # CLOSE CONNECTION
    # ---------------------------------

    def close(self):

        if self.session:

            self.session.close()

    # ---------------------------------
    # CONTEXT MANAGER
    # ---------------------------------

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        self.close()
