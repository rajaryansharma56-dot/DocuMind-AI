

\# DocuMind AI



\### Advanced AI-Powered Document Intelligence System



DocuMind AI is an intelligent document question-answering platform that allows users to upload PDF documents and interact with their content using Retrieval-Augmented Generation (RAG).



It combines semantic search, hybrid retrieval, document processing, and locally running Large Language Models to generate context-aware answers from uploaded documents.



\## Key Features



\- PDF Upload and Text Extraction

\- Intelligent Text Chunking

\- Semantic Embeddings

\- ChromaDB Vector Storage

\- Hybrid Document Retrieval

\- Cross-Encoder Reranking

\- Context-Aware Answer Generation

\- Local LLM Integration using Ollama

\- React-based User Interface

\- FastAPI Backend



\## Technology Stack



| Component | Technology |

|---|---|

| Frontend | React.js, Vite |

| Backend | Python, FastAPI |

| LLM | Ollama, Qwen3 4B |

| Embeddings | Sentence Transformers |

| Vector Database | ChromaDB |

| Retrieval | Hybrid Search |

| Reranking | Cross-Encoder |

| API | REST API |



\## System Architecture



```text

&#x20;        PDF DOCUMENT

&#x20;              |

&#x20;              v

&#x20;      PDF TEXT EXTRACTION

&#x20;              |

&#x20;              v

&#x20;       TEXT CHUNKING

&#x20;              |

&#x20;              v

&#x20;      EMBEDDING ENGINE

&#x20;              |

&#x20;              v

&#x20;      CHROMADB STORAGE

&#x20;              |

&#x20;              v

&#x20;       HYBRID RETRIEVER

&#x20;              |

&#x20;              v

&#x20;      CROSS-ENCODER

&#x20;         RERANKING

&#x20;              |

&#x20;              v

&#x20;       CONTEXT BUILDER

&#x20;              |

&#x20;              v

&#x20;       OLLAMA LLM

&#x20;          QWEN3:4B

&#x20;              |

&#x20;              v

&#x20;      AI-GENERATED ANSWER

```



\## Project Structure



```text

DocuMind-AI/

|

|-- frontend/

|   |-- public/

|   |-- src/

|   |   |-- assets/

|   |   |-- App.jsx

|   |   |-- App.css

|   |   |-- index.css

|   |   |-- main.jsx

|   |-- package.json

|

|-- src/

|   |-- ai\_engine.py

|   |-- answer\_generator.py

|   |-- context\_builder.py

|   |-- embedding\_engine.py

|   |-- embeddings.py

|   |-- generator.py

|   |-- llm.py

|   |-- pdf\_processor.py

|   |-- rag\_pipeline.py

|   |-- retriever.py

|   |-- text\_chunker.py

|   |-- vector\_store.py

|

|-- api\_server.py

|-- app.py

|-- main.py

|-- llm.py

|-- requirements.txt

|-- README.md

|-- .gitignore

```



\## Installation and Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/rajaryansharma56-dot/DocuMind-AI.git



cd DocuMind-AI

```



\### 2. Create a Virtual Environment



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 3. Install Python Dependencies



```bash

pip install -r requirements.txt

```



\### 4. Install Ollama



Download Ollama:



https://ollama.com/



Pull the required model:



```bash

ollama pull qwen3:4b

```



Make sure Ollama is running before starting the application.



\### 5. Start the Backend



From the project root:



```bash

python -m uvicorn api\_server:app --reload

```



\### 6. Start the Frontend



Open another terminal:



```bash

cd frontend



npm install



npm run dev

```



Open the local URL displayed by Vite in your browser.



\## How It Works



1\. Upload a PDF document through the frontend.

2\. Extract and preprocess its text.

3\. Divide the document into meaningful chunks.

4\. Convert chunks into vector embeddings.

5\. Store document representations in ChromaDB.

6\. Retrieve relevant information using hybrid retrieval.

7\. Rerank retrieved chunks using a Cross-Encoder.

8\. Build context from the most relevant passages.

9\. Generate answers using the local Qwen3 model.



\## Project Goals



The goal of DocuMind AI is to build a practical document intelligence system that reduces manual document searching and makes information retrieval faster and more accessible.



\## Developer



\*\*Raj Aryan Sharma\*\*



B.Tech Electronics and Instrumentation Engineering  

VIT Vellore



\- GitHub: https://github.com/rajaryansharma56-dot

\- LinkedIn: https://www.linkedin.com/in/raj-aryan-sharma-200b31412



\---



\*\*Built with Python, React, FastAPI and Generative AI.\*\*



