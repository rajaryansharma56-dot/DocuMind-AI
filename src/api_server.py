
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from functools import lru_cache

from src.rag_pipeline import RAGPipeline


app = FastAPI(
    title="DocuMind AI",
    description="AI-powered Document Intelligence Platform",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def get_rag_pipeline():
    return RAGPipeline(model="qwen3:4b")


@app.get("/")
def home():
    return {
        "project": "DocuMind AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "DocuMind AI API"
    }


@app.post("/api/chat/ask")
def ask_question(request: ChatRequest):

    try:
        rag = get_rag_pipeline()

        result = rag.ask(
            question=request.question,
            top_k=request.top_k
        )

        return {
            "success": True,
            "data": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
