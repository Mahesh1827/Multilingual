# Tirumala Multilingual AI Assistant

[![CI-CD Pipeline](https://github.com/Mahesh1827/Multilingual/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Mahesh1827/Multilingual/actions/workflows/ci-cd.yml)
[![Docker Hub](https://img.shields.io/docker/v/maheshmeesala/tirumala-ai?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/maheshmeesala/tirumala-ai)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready, **multi-agent AI Assistant** that answers pilgrim queries about Tirumala and TTD services across **5 languages** — English, Telugu, Hindi, Tamil, and Kannada — via text or voice, powered by a multilingual RAG pipeline backed by 40,000+ document chunks.

---

## ✨ Features

| Feature | Details |
|---|---|
| **5 Languages** | English, Telugu, Hindi, Tamil, Kannada — auto-detected |
| **Semantic Search** | `intfloat/multilingual-e5-large` embeddings + Qdrant vector DB |
| **Multi-Agent Pipeline** | LangGraph: FAQ Cache → Knowledge RAG → Web Search fallback |
| **Cross-Language Retrieval** | Query in Hindi, retrieve from Telugu documents — seamlessly |
| **GPU-Accelerated OCR** | PaddleOCR on CUDA for scanned PDFs |
| **LLM Providers** | Groq API (cloud, fast) → Ollama (local fallback) |
| **Voice Interface** | Whisper ASR → RAG → TTS response |
| **REST API** | Flask server, production-ready with Gunicorn |

---

## 📋 Prerequisites

Before starting, make sure you have the following installed:

| Requirement | Version | Notes |
|---|---|---|
| **Python** | `3.11.x` | Exactly 3.11 — not 3.12+ |
| **uv** | Latest | Fast package manager (`pip install uv`) |
| **Docker Desktop** | Latest | Required for Qdrant vector database |
| **NVIDIA GPU** | CUDA 12.1+ | Strongly recommended; CPU works but is very slow |
| **Ollama** | Latest | Optional — only needed if Groq API is unavailable |
| **CUDA Toolkit** | 12.1 | For GPU acceleration |

---

## 🚀 Quick Start

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Mahesh1827/Multilingual.git
cd Multilingual
```

### Step 2 — Configure Environment Variables

Copy the example environment file and fill in your API keys:

```bash
copy .env.example .env    # Windows
cp .env.example .env      # Linux/macOS
```

Edit `.env` with your keys:

```env
# Required — Get a free key at https://groq.com
GROQ_API_KEY=gsk_your_key_here

# Required — Get a free key at https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here

# Optional — For web search fallback. Get at https://tavily.com
TAVILY_API_KEY=tvly_your_key_here

# Optional overrides (defaults shown)
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 3 — Install Dependencies

```bash
# Recommended: install with uv (handles GPU torch automatically)
pip install uv
uv sync

# Alternative: standard pip (CPU torch only)
pip install -r requirements.txt
```

> **Note on GPU/PyTorch**: `uv sync` reads `pyproject.toml` and automatically pulls the correct GPU build of PyTorch (CUDA 12.1) from the official PyTorch index. If you use `pip install -r requirements.txt`, PyTorch will install without GPU support unless you install it manually first.

### Step 4 — Start the Qdrant Vector Database (Docker)

```bash
# Start Qdrant as a background Docker container with persistent storage
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 ^
    -v qdrant_storage:/qdrant/storage ^
    qdrant/qdrant
```

Verify it is running: open http://localhost:6333/dashboard in your browser.

### Step 5 — Process Documents and Build the Knowledge Base

> **Skip this step if you already have processed chunks** in `processed/extracted_text/chunk_cache/` — go directly to Step 5b.

#### Step 5a — Run the Full OCR Pipeline (first time only, ~24 hours)

This processes all PDFs in `Data_main/` using OCR and extracts text:

```bash
python main.py
```

> ⚠️ This runs OCR on 60 multilingual documents. It takes many hours on GPU. The pipeline saves progress after each document, so it is safe to interrupt and resume.

#### Step 5b — Ingest Cached Chunks into Qdrant (fast, ~30 minutes)

Once OCR is done (or if you already have the cache), push all chunks to the Qdrant Docker container:

```bash
python ingest_cache.py
```

When complete you will see:
```
✅ Migration to Qdrant Docker Complete!
```

### Step 6 — Run the AI Assistant

#### Option A: Interactive CLI (Recommended for testing)

```bash
python query_cli.py
```

Select mode 1 (Text) or 2 (Voice) and start asking questions!

#### Option B: REST API Server

```bash
# Development
flask --app server run --port 8000

# Production
gunicorn -w 1 -b 0.0.0.0:8000 server:app
```

---

## 📁 Project Structure

```
Multilingual/
│
├── Data_main/                  # Source documents (PDF/DOCX)
│   ├── English/
│   ├── Telugu/
│   ├── Hindi/
│   └── Tamil/
│
├── ocr/                        # Document Processing Pipeline
│   ├── config.py               # OCR/embedding/DB configuration
│   ├── pipeline.py             # Main OCR orchestrator
│   ├── ocr_engine.py           # PaddleOCR GPU subprocess manager
│   ├── ocr_worker.py           # Isolated OCR worker (CUDA-safe)
│   ├── extractors.py           # PDF/DOCX text & table extraction
│   └── vector_store.py         # Qdrant indexing + BM25 + reranking
│
├── query/                      # Query & Answer Pipeline
│   ├── config.py               # LLM / retrieval configuration
│   ├── agents/
│   │   ├── pipeline.py         # LangGraph orchestrator (main entry point)
│   │   ├── knowledge_rag_agent.py  # Core RAG: retrieve → reason → verify
│   │   ├── faq_agent.py        # Fast FAQ cache lookup
│   │   ├── web_agent.py        # Tavily web search fallback
│   │   ├── router_agent.py     # Domain classifier
│   │   └── validation_agent.py # Input validation & correction
│   ├── voice_pipeline.py       # Whisper STT + TTS + IndicTrans
│   ├── STT.py                  # Language detection utilities
│   └── TTS.py                  # Text-to-speech
│
├── vector_store/
│   └── metadata.json           # Chunk metadata index (40k entries)
│
├── processed/                  # Pipeline output
│   └── extracted_text/
│       └── chunk_cache/        # JSON cache — one file per document
│
├── main.py                     # Entry point: run full OCR pipeline
├── ingest_cache.py             # Push cached chunks to Qdrant Docker
├── query_cli.py                # Interactive CLI chatbot
├── server.py                   # Flask REST API
├── _env_setup.py               # Environment bootstrap (import first)
├── pyproject.toml              # uv/pip project config (GPU PyTorch index)
├── requirements.txt            # pip-compatible dependency list
├── docker-compose.yml          # Full stack: API + Ollama containers
├── Dockerfile                  # Production Docker image
└── .env                        # Your API keys (never commit this!)
```

---

## 🔌 REST API Reference

### Health Check
```http
GET /health
```
```json
{"status": "healthy", "ollama": "connected"}
```

### Query Endpoint
```http
POST /query
Content-Type: application/json

{
    "query": "How do I book darshan tickets at Tirumala?",
    "chat_history": []
}
```

**Response:**
```json
{
    "query_original": "How do I book darshan tickets at Tirumala?",
    "language": "en",
    "agent_route": "rag",
    "answer": "Darshan tickets can be booked online at ttdsevaonline.com...",
    "sources": [{"metadata": {"source_file": "TTD_dataset2.pdf", "page": 72}}],
    "verification": {"is_grounded": true, "confidence": 0.94},
    "suggestions": ["What is the dress code for darshan?", ...],
    "response_time_s": 2.34
}
```

### Example — Query in Telugu
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "తిరుమలలో దర్శన టికెట్ ఎలా బుక్ చేయాలి?"}'
```

---

## ⚙️ Configuration Reference

### `ocr/config.py` — Document Processing

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-large` | Multilingual embedding model |
| `EMBEDDING_DEVICE` | `cuda` / `cpu` | Auto-detected from GPU availability |
| `QDRANT_URL` | `http://localhost` | Qdrant Docker host |
| `QDRANT_PORT` | `6333` | Qdrant Docker port |
| `QDRANT_COLLECTION_NAME` | `tirumala_multilingual` | Collection name |
| `CHUNK_SIZE` | `600` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `PADDLEOCR_USE_GPU` | `True` | Enable GPU OCR |

### `query/config.py` — Query Pipeline

| Setting | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | from `.env` | Groq cloud API key (set to enable) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Best multilingual LLM on Groq |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Best local fallback for Indic languages |
| `TOP_K_RETRIEVE` | `6` | Chunks retrieved per query |
| `TOP_K_FINAL` | `5` | Chunks passed to LLM after reranking |

---

## 🌐 Language Support

The system automatically detects and processes queries in all 5 languages:

| Language | Script | Detection Method |
|---|---|---|
| English | Latin | ASCII ratio + Lingua |
| Telugu | Telugu script (U+0C00–U+0C7F) | Unicode range detection |
| Hindi | Devanagari (U+0900–U+097F) | Unicode range detection |
| Tamil | Tamil script (U+0B80–U+0BFF) | Unicode range detection |
| Kannada | Kannada script (U+0C80–U+0CFF) | Unicode range detection |
| Romanized Indic | Latin | Vocabulary/phonetic patterns |

---

## 🏗️ Agent Pipeline Flow

```
User Query
    │
    ▼
Language Detection → Translation to English (if needed)
    │
    ▼
[LangGraph Pipeline]
    │
    ├─ Validate & Correct Query
    │
    ├─ Classify Intent (Factual / Real-time / Conversational)
    │       │
    │       ├─ Conversational → FAQ Cache (instant)
    │       │
    │       ├─ Factual ──────► FAQ Cache
    │       │                      │ miss
    │       │                      ▼
    │       │                  Knowledge RAG
    │       │                  (Qdrant + Reranker + LLM)
    │       │                      │ no answer
    │       │                      ▼
    │       └─ Real-time ────► Web Search (Tavily)
    │
    ▼
Answer Verification (groundedness check)
    │
    ▼
Translation to user's language (if needed)
    │
    ▼
Final Answer + Sources + Follow-up Suggestions
```

---

## 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'langchain_qdrant'`
```bash
uv pip install langchain-qdrant
# or
pip install langchain-qdrant
```

### ❌ `Connection refused` when connecting to Qdrant
Make sure Docker Desktop is running and the Qdrant container is up:
```bash
docker ps               # check if 'qdrant' container is running
docker start qdrant     # start it if stopped
```

### ❌ Ollama not found / not running
Ollama is only needed if `GROQ_API_KEY` is not set. If you have a Groq key in `.env`, Ollama is not required. To install Ollama: https://ollama.com

### ❌ CUDA / GPU not detected
```python
# Verify in Python:
import torch
print(torch.cuda.is_available())  # should print True
print(torch.cuda.get_device_name(0))
```
Ensure CUDA 12.1 Toolkit is installed and you used `uv sync` (not `pip install -r requirements.txt`) for the GPU version of PyTorch.

### ❌ OCR pipeline freezes on large PDFs
This is expected for very large books (>60 pages). The pipeline automatically skips table extraction on large documents. Each document is cached individually, so you can safely restart `python main.py` and it will resume from where it left off.

### ❌ `ingest_cache.py` — no chunks found
Make sure the OCR pipeline has completed at least some documents:
```bash
# Check how many files are cached:
ls processed/extracted_text/chunk_cache/*.json | wc -l   # Linux
dir processed\extracted_text\chunk_cache\*.json /B | find /c /v ""  # Windows
```

### ❌ Answer quality is poor / hallucinations
- Ensure Groq API key is valid (fastest, best quality)
- Check Qdrant has all 40,000 points: http://localhost:6333/dashboard
- Try rephrasing the query — the pipeline auto-corrects ASR errors but may miss some

---

## 🛠 CI/CD Pipeline

| Stage | Trigger | Action |
|-------|---------|--------|
| **Lint & Test** | Every push/PR | flake8 + pytest with coverage |
| **Docker Build** | Push to `main` | Build image → push to Docker Hub |
| **Deploy** *(disabled)* | Push to `main` | SSH deploy to production server |

To enable production deploy, add these GitHub Secrets:
`SSH_HOST`, `SSH_USER`, `SSH_PRIVATE_KEY`, `SSH_PORT`

---

## 🙏 Acknowledgements

- **Tirumala Tirupati Devasthanams (TTD)** — for the source documents
- **intfloat/multilingual-e5-large** — multilingual embedding model
- **Qdrant** — high-performance vector database
- **PaddleOCR** — multilingual GPU OCR
- **LangChain + LangGraph** — agent orchestration framework
- **Groq** — ultra-fast LLM inference
