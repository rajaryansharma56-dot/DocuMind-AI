
from llm import OllamaLLM


class AnswerGenerator:

    def __init__(self):
        self.llm = OllamaLLM(model="qwen3:4b")

    def generate_answer(self, question, retrieved_results):

        if not retrieved_results:
            return "No relevant information found in your documents."

        context_parts = []

        for index, result in enumerate(retrieved_results, start=1):

            filename = result.get("filename", "Unknown document")
            page = result.get("page_number", "Unknown")
            content = result.get("content", "")

            context_parts.append(
                f"""
SOURCE {index}
Document: {filename}
Page: {page}

Content:
{content}
"""
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are answering a question using retrieved PDF content.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

STRICT INSTRUCTIONS:

1. Give only the final answer.
2. Never display internal analysis or thinking.
3. Use only information supported by the document.
4. Do not invent facts or missing information.
5. If the answer is unavailable, clearly say so.
6. Use clear headings and bullet points when useful.
7. Include source document names and page numbers.
8. Avoid repeating the same information.
9. Answer the exact question directly.

FINAL ANSWER:
"""

        system_prompt = """
You are DocuMind AI, a professional document
understanding assistant.

Provide accurate, concise and well-structured answers.
Use retrieved document content as your source of truth.
Never fabricate information or citations.
Return only the final response.
"""

        return self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
