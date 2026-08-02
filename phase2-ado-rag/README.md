# RAG Project — API

Retrieval-Augmented Generation API that ingests documents, links, and messages; indexes them in a vector store; and answers natural-language questions with grounded, cited responses and **zero hallucinated answers**.

This is the **Phase 2** implementation of the SES RAG system. It runs a full REST API (FastAPI), an interactive Streamlit UI, and optional Azure DevOps integration with an automatic Ragas evaluation pipeline.

> Full project overview (architecture, flow, stack, structure, endpoints): see the [root README](../README.md).

---

## Tech Stack

| Component | Technology |
|:---|:---|
| **LLM** | **Groq** (cloud, default — `llama-3.3-70b-versatile`) or **Ollama** (local — `qwen2.5:14b`) |
| **Embeddings** | bge-small-en-v1.5 via sentence-transformers (local) |
| **Orchestration** | LlamaIndex |
| **Vector DB** | ChromaDB (persistent) |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Evaluation** | Ragas |
| **Doc Parsing** | PyMuPDF, python-docx, Trafilatura |
| **Config** | Pydantic Settings + `.env` |

---

## Features

- **Multi-format ingestion** — PDF, DOCX, web URLs, pasted text/messages.
- **Content-type-aware chunking** — PDF (450), URL (350), chat (200), ADO (400) tokens.
- **Session-scoped RAG** — create and manage chat sessions; scope ingestion + retrieval per session.
- **Relevance grading** — filters irrelevant context and **declines low-confidence answers** (zero hallucination).
- **HyDE retrieval** — Hypothetical Document Embeddings for better recall.
- **Two output modes** — `qa` (free-text with citations) and `test_case` (structured QA test cases, auto-detected for ADO items).
- **Grounded citations** — answers cite sources `[Source: <title>]`.
- **Ragas evaluation** — faithfulness, relevancy, context precision/recall on golden Q&A sets.
- **Switchable LLM** — cloud (Groq) or local (Ollama) via `.env`.

---

## Quick Start

```bash
# 1. Set up
cd phase2-ado-rag
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate            # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (copy the SAFE template — never commit the .env itself)
cp .env.example .env
# Fill in GROQ_API_KEY (or set LLM_PROVIDER=ollama for local)

# 4. Generate synthetic data & build the index
python scripts/generate_synthetic_data.py
python scripts/build_index.py

# 5. Start the API server (port 8001)
uvicorn src.ragpoc.api.main:app --reload --port 8001

# 6. Start the Streamlit UI in a separate terminal (port 8502)
streamlit run ui/app.py --server.port 8502
```

API docs: http://localhost:8001/docs · UI: http://localhost:8502

---

## Prerequisites

1. **Python 3.11+**
2. **Groq API key** (default) from [console.groq.com](https://console.groq.com/) — or
3. **Ollama** (local option): `ollama pull qwen2.5:14b`

---

## Project Structure

```
phase2-ado-rag/
├── config/settings.py          # Central configuration (env-driven)
├── src/ragpoc/
│   ├── ingestion/              # loaders, chunker, normalizer, mcp_client
│   ├── models/                 # groq_llm, llm(ollama), embeddings, registry
│   ├── retrieval/              # vector_store, retriever, relevance_grader
│   ├── generation/             # prompt_templates, pipeline (retrieve→grade→generate)
│   ├── evaluation/             # golden_set, ragas_runner
│   └── api/                    # main + routes: ingest, query, session, evaluate
├── ui/app.py                   # Streamlit demo UI
├── scripts/                    # build_index, generate_synthetic_data, ADO utils
├── tests/                      # Unit tests
├── data/                       # Synthetic docs & golden Q&A set
├── requirements.txt
├── .env.example                # Safe config template
```

---

## API Endpoints (port 8001)

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | Provider health + vector-store stats |
| `POST` | `/ingest/file` | Upload PDF/DOCX |
| `POST` | `/ingest/url` | Scrape + ingest a URL |
| `POST` | `/ingest/message` | Ingest pasted text |
| `GET` | `/ingest/status` | Ingestion stats (per session) |
| `POST` | `/query` | Ask a question → grounded answer |
| `GET` | `/query/health` | Query pipeline readiness |
| `POST` | `/evaluate/run` | Trigger Ragas evaluation |
| `GET` | `/evaluate/results` | List past eval results |
| `GET` | `/evaluate/results/latest` | Latest eval JSON |
| `POST` | `/session` | Create a session |
| `GET` | `/session` | List sessions |
| `GET` | `/session/{id}` | Session details |
| `GET` | `/session/{id}/stats` | Session stats |
| `DELETE` | `/session/{id}` | Delete a session |

---

## Evaluation

```bash
python -m src.ragpoc.evaluation.ragas_runner
# or via API: POST /evaluate/run
```

Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall.

---

## Security

- **Never commit `.env`** — it holds real API keys (Groq, ADO PAT).
- Share the **`.env.example`** instead.
- ChromaDB `storage/` and `__pycache__/` are git-ignored.