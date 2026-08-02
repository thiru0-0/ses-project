# SES RAG Project

**SES (Software Engineering Support) RAG** — A Retrieval-Augmented Generation system for software engineering knowledge management. It ingests software engineering artifacts (requirements, specs, design docs, chat logs, code reviews) and answers natural-language questions with **grounded, cited responses**.

> **Goal**: Achieve **≥80% answer accuracy with zero hallucinated answers** on software-engineering Q&A.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [LLM Provider Options](#llm-provider-options)
- [How to Run](#how-to-run)
- [Evaluation](#evaluation)
- [Testing & Code Quality](#testing--code-quality)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The project is built in **two progressive phases**, both sharing the same RAG core:

| Phase | Directory | Focus |
|:------|:----------|:------|
| **Phase 1** | `phase1-rag-poc/` | Core RAG over local documents (PDF, DOCX, URLs, chat exports) |
| **Phase 2** | `phase2-ado-rag/` | Adds cloud LLM (Groq), Azure DevOps sources, **query sessions API**, and HyDE retrieval |

Phase 2 is the primary, up-to-date API implementation and is the focus of this documentation.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SES RAG PIPELINE                                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   CALLER (Streamlit UI / HTTP API)                                     │
│        │                                                               │
│        ▼                                                               │
│  ┌───────────────┐        ┌────────────────────────────────────────┐  │
│  │   INGESTION   │        │            GENERATION                   │  │
│  │               │        │                                        │  │
│  │ • PDF / DOCX  │        │  query ─► retrieve ─► grade ─►         │  │
│  │ • Web URLs    │        │  (HyDE)      (vector   (relevance       │  │
│  │ • Pasted text │        │              search)   grader)         │  │
│  │ • Chat logs   │        │                      │                 │  │
│  │ • ADO items   │        │        ┌─────────┐   ▼                 │  │
│  │               │        │        │ RETRIEVAL│ generate / decline │  │
│  └──────┬────────┘        └───────►│ ChromaDB  ──► answer            │  │
│         │ normalize · chunk        │ HyDE      ◄─ embeddings        │  │
│         │                          └─────────┘                     │  │
└─────────┴────────────────────────────────────────────────────────────┘
```

**Key components:**
- **Ingestion** — Loads & parses multiple source types, normalizes text, splits into variable-size chunks (content-type aware).
- **Vector Store** — [ChromaDB](https://www.trychroma.com/) persistent embedding store for similarity search, optional per-session scoping.
- **Retrieval** — Top-k vector search with **HyDE** (Hypothetical Document Embeddings) query expansion.
- **Relevance Grader** — Filters retrieved chunks by a confidence threshold; declines to answer if nothing is relevant (prevents hallucinations).
- **Generation** — LLM produces grounded answers from relevant chunks with `[Source: <title>]` citations, in either `qa` or `test_case` mode.

---

## Data Flow

A successful query travels through four stages:

```
1. RETRIEVE   ▸ Top-k chunks fetched from ChromaDB (vector similarity + optional HyDE expansion)
      │
2. GRADE      ▸ Relevance grader scores each chunk against the question
      │        ▸ If NO chunk passes the threshold → answer is DECLINED (no hallucination)
      ▼
3. GENERATE   ▸ LLM builds grounded answer from the relevant chunks only
      │        ▸ Sources are appended as citations [Source: <title>]
      ▼
4. RESPOND    ▸ { answer, sources[], confidence, declined, retrieved_chunks, relevant_chunks }
```

**Output modes** (selected via `mode` or auto-detected from source type):
- **`qa`** — free-text answer with citations (default).
- **`test_case`** — structured QA test-case output (auto-chosen for ADO work items / wiki).

---

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| **LLM Provider** | **Groq** (cloud, default — `llama-3.3-70b-versatile`) or **Ollama** (local — `qwen2.5:14b`) |
| **Embeddings** | BAAI/bge-small-en-v1.5 via [sentence-transformers](https://www.sbert.net/) (local) |
| **Orchestration** | [LlamaIndex](https://www.llamaindex.ai/) |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) (persistent) |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **UI** | [Streamlit](https://streamlit.io/) |
| **Evaluation** | [Ragas](https://docs.ragas.io/) |
| **Config** | Pydantic Settings + `.env` |
| **Doc Parsing** | PyMuPDF, python-docx, Trafilatura |
| **Validation** | Pydantic v2, pytest |

---

## Features

- **Multi-format ingestion** — PDF, DOCX, web URLs (boilerplate removal via Trafilatura), pasted text/messages.
- **Content-type-aware chunking** — separate token/chunk sizes for PDF (450), URL (350), chat (200), and ADO (400).
- **Session-scoped RAG** — create chat sessions and scope ingestion + retrieval per session.
- **Relevance grading** — filters irrelevant context and **declines low-confidence answers** to avoid hallucination.
- **Grounded citations** — every answer cites sources `[Source: <title>]`.
- **HyDE retrieval** — hypothetical-document expansion for better recall on technical questions.
- **Test-case mode** — auto-generates structured QA test cases from ADO user stories.
- **Automatic evaluation** — Ragas metrics (faithfulness, relevancy, context precision/recall) on golden Q&A sets.
- **Cloud or local LLM** — switch instantaneously via `.env`.
- **Interactive UI + REST API** — Streamlit UI plus full OpenAPI docs at `/docs`.

---

## Project Structure

```
ses-project/
├── .gitignore               # Excludes secrets, caches, vector DB
├── README.md                 # This file
│
├── phase1-rag-poc/           # Phase 1 · Core RAG POC
│   ├── config/settings.py
│   ├── src/ragpoc/
│   │   ├── ingestion/        # loaders, chunker, normalizer
│   │   ├── models/           # LLM & embedding abstraction
│   │   ├── retrieval/        # vector store, retriever, grader
│   │   ├── generation/       # prompt templates, pipeline
│   │   ├── evaluation/       # golden set, Ragas runner
│   │   └── api/              # FastAPI routes
│   ├── ui/app.py             # Streamlit demo UI
│   ├── scripts/ tests/ data/ ./env.example  README.md
│
└── phase2-ado-rag/            # Phase 2 · Full RAG API (primary)
    ├── config/settings.py             # Central config (env-driven)
    ├── src/ragpoc/
    │   ├── ingestion/                 # loaders, chunker, normalizer, mcp_client
    │   ├── models/                    # groq_llm, llm, embeddings, registry, base
    │   ├── retrieval/                 # vector_store, retriever, relevance_grader
    │   ├── generation/                # pipeline, prompt_templates
    │   ├── evaluation/                # ragas_runner, golden_set
    │   └── api/                       # main, routes_ingest, routes_query, routes_session, routes_evaluate
    ├── ui/app.py                      # Streamlit UI
    ├── scripts/                       # build_index, generate_synthetic_data, ADO utils
    ├── tests/                         # unit tests
    ├── data/                          # synthetic docs, golden QA, eval results
    ├── requirements.txt
    ├── .env.example                   # safe config template (share this)
    └── README.md                      # Phase 2 details
```

---

## API Endpoints

The FastAPI server runs on **port 8001**. Interactive docs: `http://localhost:8001/docs`

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | Overall health: LLM provider status, embedding model, vector-store stats |
| `POST` | `/ingest/file` | Upload a **PDF or DOCX** (multipart) for ingestion |
| `POST` | `/ingest/url` | Scrape + ingest a **web URL** (`{ "url": ... }`) |
| `POST` | `/ingest/message` | Ingest pasted **text** (`{ "text": ... }`) |
| `GET` | `/ingest/status` | Ingestion / collection statistics (optional `session_id`) |
| `POST` | `/query` | Ask a question → grounded answer with sources `{ "question": ... }` |
| `GET` | `/query/health` | Query pipeline readiness check |
| `POST` | `/evaluate/run` | Trigger a blocking Ragas evaluation run |
| `GET` | `/evaluate/results` | List past evaluation results with metrics |
| `GET` | `/evaluate/results/latest` | Fetch the most recent evaluation JSON |
| `POST` | `/session` | Create a new chat session `{ "name": ... }` |
| `GET` | `/session` | List all sessions with chunk counts |
| `GET` | `/session/{id}` | Get session details |
| `GET` | `/session/{id}/stats` | Statistics for a specific session |
| `DELETE` | `/session/{id}` | Delete a session and its data |

---

## LLM Provider Options

Both phases support **two LLM backends** — switch via `.env`:

| Provider | Type | Model | Setup | Cost | Privacy |
|:---------|:-----|:------|:------|:-----|:--------|
| **Groq** *(default)* | Cloud | `llama-3.3-70b-versatile` | API key at groq.com | Free-tier / pay-per-token | Sends data to cloud |
| **Ollama** | Local | `qwen2.5:14b` | `ollama pull qwen2.5:14b` | Free (local hardware) | 100% local |

```env
# Cloud (default)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile

# Local
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:14b
```

---

## How to Run

### Prerequisites
- **Python 3.11+**
- For **Groq** (default): a free API key at [groq.com](https://console.groq.com/)
- For **Ollama** (optional): install from [ollama.com](https://ollama.com)

### Setup
```bash
# 1. Clone the repo
git clone https://github.com/thiru0-0/ses-project
cd ses-project/phase2-ado-rag

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your private config from the safe template
cp .env.example .env
# Fill in your GROQ_API_KEY (or switch to Ollama). NEVER commit .env.

# 5. Generate synthetic data & build the vector index
python scripts/generate_synthetic_data.py
python scripts/build_index.py

# 6. Start the API server (port 8001)
uvicorn src.ragpoc.api.main:app --reload --port 8001

# 7. Start the Streamlit UI in a separate terminal (port 8502)
streamlit run ui/app.py --server.port 8502
```

Then:
- **API docs**: http://localhost:8001/docs
- **Streamlit UI**: http://localhost:8502

### Ports
| Service | Port |
|---------|------|
| API | 8001 |
| Streamlit UI | 8502 |

---

## Evaluation

Run Ragas evaluation against the golden Q&A set to measure quality:

```bash
# Running in phase2
cd phase2-ado-rag
python scripts/run_ado_evaluation.py          # phase-2 styled eval
# or via the API:
# POST http://localhost:8001/evaluate/run

# Phase 1
cd phase1-rag-poc
python -m src.ragpoc.evaluation.ragas_runner
```

**Metrics reported**: Faithfulness, Answer Relevancy, Context Precision, Context Recall — goal is **≥80% accuracy, zero hallucinations**.

---

## Testing & Code Quality

```bash
cd phase2-ado-rag
pytest tests/                    # run unit tests
ruff check src/ tests/           # lint (optional)
black src/ tests/                # format (optional)
mypy src/                        # type-check (optional)
```

---

## Security

- **Never commit `.env`** — it holds real API keys (Groq, Azure DevOps). The included `.gitignore` excludes it.
- Share the **`.env.example`** template instead — it carries no real secrets.
- If a key is ever exposed, **revoke it immediately** and generate a new one.
- ChromaDB `storage/` and `__pycache__/` are also git-ignored (runtime data, not source).

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Make changes with tests.
4. Run lint/type-check: `ruff check src/ && mypy src/`.
5. Submit a pull request.

---

## License

MIT License — see `LICENSE` file for details.