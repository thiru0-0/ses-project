# RAG POC — Phase 2: ADO Test Case Assistant

Retrieval-Augmented Generation proof-of-concept that acts as an automated test case generation assistant. Users can upload documents, share links, or paste messages into a workspace; the system parses & indexes that content; and answers natural-language questions or generates structured test cases from Azure DevOps (ADO) user stories with high accuracy and zero hallucination.

## Tech Stack

| Component | Technology |
|:---|:---|
| **LLM** | Qwen2.5:14B-Instruct via Ollama (fully local) |
| **Embeddings** | bge-small-en-v1.5 via sentence-transformers (local) |
| **Orchestration** | LlamaIndex |
| **Vector DB** | ChromaDB (persistent, session-filtered) |
| **Retrieval** | Hypothetical Document Embeddings (HyDE) |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Evaluation** | Custom 5-dimension scoring & Ragas |
| **Doc Parsing** | PyMuPDF, python-docx, Trafilatura, custom ADO JSON loader |

## Key Phase 2 Features

- **Test Case Generation**: Dynamically formats output into a structured 8-field table (Test ID, Requirement Ref, Title, Preconditions, Steps, Expected Result, Actual Result, Status) when ADO work items are detected in the context.
- **Session Scoping**: Queries can be scoped to filter the vector store by the current user's session, enabling isolated single-instance workflows without needing multiple databases.
- **Unified Smart Ingestion**: A single endpoint (`/ingest/auto`) handles document uploads, URL scraping, and pasted text dynamically.
- **Human-in-the-loop Guardrails**: Automatically detects low-confidence responses and flags them for human review directly in the UI.
- **Advanced Evaluation**: Evaluates ADO responses across 5 dimensions: Test Coverage, Traceability, Faithfulness, Structural Completeness, and Guardrail Behavior.

## Prerequisites

1. **Python 3.11+**
2. **Ollama** — install from [ollama.com](https://ollama.com) and pull the model:
   ```bash
   ollama pull qwen2.5:14b
   ```

## Quick Start

```bash
# 1. Clone and set up environment
cd phase2-ado-rag
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
uvicorn src.ragpoc.api.main:app --reload --port 8001

# 6. Start the Streamlit UI (separate terminal)
streamlit run ui/app.py --server.port 8502
```

## Project Structure

```
phase2-ado-rag/
├── config/settings.py          # Central configuration
├── src/ragpoc/
│   ├── ingestion/              # Document loading, ADO loader, normalization, chunking
│   ├── models/                 # LLM & embedding provider abstraction
│   ├── retrieval/              # Vector store, retriever (HyDE), relevance grading
│   ├── generation/             # Prompt templates, RAG pipeline orchestrator
│   ├── evaluation/             # Golden set loader, 5-dimension ADO evaluation runner
│   └── api/                    # FastAPI endpoints
├── ui/app.py                   # Streamlit demo UI (2-tab layout)
├── scripts/                    # One-off utilities (synthetic data, index builder)
├── tests/                      # Unit tests
└── data/                       # Synthetic docs, ADO items, golden Q&A set
```

## Content-Type Chunk Sizes

| Content Type | Chunk Size | Overlap |
|:---|:---|:---|
| PDF / DOCX | 450 tokens | 80 tokens |
| Scraped URLs | 350 tokens | 60 tokens |
| ADO Work Items | 400 tokens | 70 tokens |
| Chat exports | 200 tokens | 30 tokens |
