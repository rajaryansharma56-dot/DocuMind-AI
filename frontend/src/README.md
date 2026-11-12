# DocuMind AI 🧠
### Advanced AI-Powered Document Intelligence System

DocuMind AI is an intelligent document question-answering platform built using Retrieval-Augmented Generation (RAG). It allows users to upload PDF documents, retrieve relevant information, and receive context-aware answers using a locally running Large Language Model.

## 🚀 Key Features

- **PDF Processing:** Extract and process text from PDF documents.
- **Intelligent Chunking:** Divide documents into meaningful text chunks.
- **Vector Database:** Store document embeddings using ChromaDB.
- **Hybrid Retrieval:** Retrieve relevant information using multiple retrieval techniques.
- **Reranking:** Improve retrieval relevance using a Cross-Encoder.
- **AI Answer Generation:** Generate contextual responses using Qwen3:4B.
- **Local AI Processing:** Run the language model through Ollama.
- **Interactive Interface:** Upload PDFs and ask questions through a React frontend.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite |
| Backend | Python, FastAPI |
| LLM | Qwen3:4B |
| LLM Runtime | Ollama |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Reranking | Cross-Encoder |
| API Server | Uvicorn |

## 🏗️ System Architecture

```text
             USER
               |
               v
       REACT FRONTEND
               |
               v
        FASTAPI BACKEND
               |
       +-------+-------+
       |               |
       v               v
   PDF UPLOAD      USER QUERY
       |               |
       v               v
 TEXT EXTRACTION   HYBRID RETRIEVER
       |               |
       v               v
    CHUNKING       CROSS-ENCODER
       |               |
       v               v
  EMBEDDINGS      RELEVANT CHUNKS
       |               |
       v               v
    CHROMADB          |
                       |
                       v
                QWEN3:4B LLM
                       |
                       v
                 AI RESPONSE
                       |
                       v
                  USER
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Backend setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

Install Ollama and download the model:

```bash
ollama pull qwen3:4b
```

Make sure the Ollama service is running.

### 4. Start the backend

```bash
python -m uvicorn api_server:app --reload
```

### 5. Frontend setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the localhost URL displayed by Vite.

## 💡 How It Works

1. Upload a PDF document.
2. DocuMind extracts its text.
3. Extracted text is divided into smaller chunks.
4. Embeddings are generated and stored in ChromaDB.
5. A user submits a question.
6. Hybrid retrieval identifies relevant document chunks.
7. A Cross-Encoder reranks the retrieved results.
8. Qwen3:4B generates an answer using the retrieved context.

## 🎯 Project Objective

The objective of DocuMind AI is to make information retrieval from large documents faster, more accessible, and more intelligent through locally running AI models and Retrieval-Augmented Generation.

## 🔮 Future Improvements

- Multiple document management.
- Source citations and page references.
- Conversation history.
- User authentication.
- Cloud deployment.
- Advanced document analytics.
- Support for additional document formats.

## 👨‍💻 Developer

**Raj Aryan Sharma**

Electronics and Instrumentation Engineering

**Interests:** Artificial Intelligence | Backend Development | Machine Learning | RAG Systems | Software Engineering

---

*DocuMind AI — Turning documents into knowledge.*