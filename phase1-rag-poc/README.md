# RAG POC

Retrieval-Augmented Generation proof-of-concept that lets users upload documents, share links, or paste messages into a workspace; parses & indexes that content; and answers natural-language questions with ≥ 80% accuracy and zero hallucinated answers.

## Tech Stack

| Component | Technology |
|:---|:---|
| **LLM** | Qwen2.5:14B-Instruct via Ollama (fully local) |
| **Embeddings** | bge-small-en-v1.5 via sentence-transformers (local) |
| **Orchestration** | LlamaIndex |
| **Vector DB** | ChromaDB (persistent) |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Evaluation** | Ragas |
| **Doc Parsing** | PyMuPDF, python-docx, Trafilatura |

## Prerequisites

1. **Python 3.11+**
2. **Ollama** — install from [ollama.com](https://ollama.com) and pull the model:
   ```bash
   ollama pull qwen2.5:14b
   ```

## Quick Start

```bash
# 1. Clone and set up environment
cd rag-poc
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env if needed (defaults work for local development)

# 4. Generate synthetic data & build index
python scripts/generate_synthetic_data.py
python scripts/build_index.py

# 5. Start the API server
uvicorn src.ragpoc.api.main:app --reload --port 8000

# 6. Start the Streamlit UI (separate terminal)
streamlit run ui/app.py
```

## Project Structure

```
rag-poc/
├── config/settings.py          # Central configuration
├── src/ragpoc/
│   ├── ingestion/              # Document loading, normalization, chunking
│   ├── models/                 # LLM & embedding provider abstraction
│   ├── retrieval/              # Vector store, retriever, relevance grading
│   ├── generation/             # Prompt templates, RAG pipeline orchestrator
│   ├── evaluation/             # Golden set loader, Ragas evaluation runner
│   └── api/                    # FastAPI endpoints
├── ui/app.py                   # Streamlit demo UI
├── scripts/                    # One-off utilities
├── tests/                      # Unit tests
└── data/                       # Synthetic docs & golden Q&A set
```

## Content-Type Chunk Sizes

| Content Type | Chunk Size | Overlap |
|:---|:---|:---|
| PDF / DOCX | 450 tokens | 80 tokens |
| Scraped URLs | 350 tokens | 60 tokens |
| Chat exports | 200 tokens | 30 tokens |
