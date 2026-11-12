
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGPipeline


# ==========================================
# DOCUMIND AI - FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="DocuMind AI",
    description="AI-Powered Document Intelligence Platform",
    version="1.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# REQUEST VALIDATION
# ==========================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=5000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


# ==========================================
# RAG PIPELINE INITIALIZATION
# ==========================================

@lru_cache(maxsize=1)
def get_rag_pipeline():

    return RAGPipeline(
        model="qwen3:4b"
    )


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "project": "DocuMind AI",
        "description": "AI-Powered Document Intelligence",
        "version": "1.0.0",
        "status": "running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "DocuMind AI API"
    }


# ==========================================
# CHAT ENDPOINT
# ==========================================

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
            detail=f"AI processing failed: {str(error)}"
        )
