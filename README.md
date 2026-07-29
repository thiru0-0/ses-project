# SES RAG Project

**SES (Software Engineering Support) RAG** — A Retrieval-Augmented Generation system for software engineering knowledge management, with two progressive phases.

---

## Project Overview

This project implements a RAG pipeline that ingests software engineering artifacts (requirements, specs, chat logs, code reviews, Azure DevOps work items) and answers natural-language questions with grounded, cited responses.

**Goal**: Achieve ≥80% answer accuracy with zero hallucinations on software engineering Q&A.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SES RAG Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  Ingestion          │  Retrieval          │  Generation         │
│  ─────────────────  │  ─────────────────  │  ─────────────────  │
│  • PDF/DOCX         │  • ChromaDB         │  • LLM Provider     │
│  • Web URLs         │  • Vector Search    │    - Ollama (local) │
│  • Chat Exports     │  • Relevance Grade  │    - OpenRouter     │
│  • Azure DevOps     │  • HyDE (Phase 2)   │  • Prompt Templates │
│    (Phase 2)        │                     │  • RAG Pipeline     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Qwen2.5:14B via Ollama (local) **or** OpenRouter (cloud) |
| **Embeddings** | BAAI/bge-small-en-v1.5 via sentence-transformers (local) |
| **Orchestration** | LlamaIndex |
| **Vector DB** | ChromaDB (persistent) |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Evaluation** | Ragas |
| **Doc Parsing** | PyMuPDF, python-docx, Trafilatura |
| **Config** | Pydantic Settings + .env |

---

## Project Structure

```
ses-project/
├── phase1-rag-poc/          # Phase 1: Core RAG POC
│   ├── config/              # Settings & configuration
│   ├── src/ragpoc/          # Core RAG modules
│   │   ├── ingestion/       # Document loading, chunking, normalization
│   │   ├── models/          # LLM & Embedding provider abstraction
│   │   ├── retrieval/       # Vector store, retriever, grader
│   │   ├── generation/      # Prompt templates, RAG pipeline
│   │   ├── evaluation/      # Golden set, Ragas runner
│   │   └── api/             # FastAPI endpoints
│   ├── ui/app.py            # Streamlit demo UI
│   ├── scripts/             # Data generation, index building
│   ├── tests/               # Unit tests
│   ├── data/                # Synthetic docs, golden QA, eval results
│   ├── storage/             # ChromaDB persistence
│   ├── INSTRUCTIONS.md      # LLM setup guide (local + OpenRouter)
│   ├── .env.example         # Environment template
│   └── README.md            # Phase 1 details
│
├── phase2-ado-rag/          # Phase 2: Azure DevOps Integration
│   ├── config/              # Extended settings
│   ├── src/ragpoc/          # Extended modules
│   │   ├── ingestion/       # + ADO loaders (work items, PRs, wiki)
│   │   ├── retrieval/       # + HyDE retriever
│   │   └── api/             # + ADO routes
│   ├── scripts/             # + ADO ingestion, evaluation
│   ├── data/evaluation/     # ADO golden set
│   ├── INSTRUCTIONS.md      # Phase 2 specific setup
│   ├── .env.example         # Environment template (+ ADO config)
│   └── README.md            # Phase 2 details
│
└── README.md                # This file
```

---

## Phase 1: Core RAG POC

**`phase1-rag-poc/`** — Standalone RAG system for software engineering documents.

### Capabilities
- Ingest PDF, DOCX, web URLs, chat exports (Slack/Teams/JSON)
- Chunk with content-type-aware sizes (PDF: 450, URL: 350, Chat: 200 tokens)
- Vector search with relevance grading
- Grounded generation with citations `[Source: <title>]`
- Ragas evaluation on golden Q&A set
- Streamlit UI for interactive Q&A

### Quick Start
```bash
cd phase1-rag-poc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: choose LLM_PROVIDER=ollama or openrouter
python scripts/generate_synthetic_data.py
python scripts/build_index.py
uvicorn src.ragpoc.api.main:app --reload --port 8000
streamlit run ui/app.py  # separate terminal
```

### Documentation
- **Setup Guide**: `phase1-rag-poc/INSTRUCTIONS.md`
- **Details**: `phase1-rag-poc/README.md`

### Ports
| Service | Port |
|---------|------|
| API | 8000 |
| Streamlit UI | 8501 |

---

## Phase 2: Azure DevOps Integration

**`phase2-ado-rag/`** — Extends Phase 1 with Azure DevOps data sources and advanced retrieval.

### New Capabilities
- **ADO Ingestion**: Work items, pull requests, wiki pages, commits
- **HyDE Retrieval**: Hypothetical Document Embeddings for better recall on technical queries
- **ADO-Specific Chunking**: 400 tokens / 70 overlap for work items
- **Extended Evaluation**: ADO-specific golden QA set with Ragas
- **API Extensions**: `/ingest/ado`, `/query/ado` endpoints

### Quick Start
```bash
cd phase2-ado-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add ADO_ORG_URL, ADO_PROJECT, ADO_PAT
python scripts/generate_synthetic_data.py
python scripts/build_index.py
uvicorn src.ragpoc.api.main:app --reload --port 8001
streamlit run ui/app.py --server.port 8502
```

### Documentation
- **Setup Guide**: `phase2-ado-rag/INSTRUCTIONS.md`
- **Details**: `phase2-ado-rag/README.md`

### Ports
| Service | Port |
|---------|------|
| API | 8001 |
| Streamlit UI | 8502 |

---

## LLM Provider Options

Both phases support **two LLM backends** — choose per environment:

| Provider | Type | Setup | Cost | Privacy |
|----------|------|-------|------|---------|
| **Ollama** | Local | `ollama pull qwen2.5:14b` | Free (hardware) | 100% local |
| **OpenRouter** | Cloud | API key at openrouter.ai | Pay-per-token | Varies by model |

**Switch instantly** via `.env`:
```env
# Local
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:14b

# Cloud
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openai/gpt-4o-mini
```

See `phase1-rag-poc/INSTRUCTIONS.md` for detailed setup, model recommendations, and cost estimates.

---

## Evaluation

Both phases include Ragas-based evaluation:

```bash
# Phase 1
cd phase1-rag-poc
python -m src.ragpoc.evaluation.ragas_runner

# Phase 2 (includes ADO golden set)
cd phase2-ado-rag
python -m src.ragpoc.evaluation.ragas_runner
python scripts/run_ado_evaluation.py
```

Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall

---

## Development

### Run Tests
```bash
cd phase1-rag-poc && pytest tests/
cd phase2-ado-rag && pytest tests/
```

### Code Quality
```bash
# Both phases use same stack
pip install ruff black mypy
ruff check src/ tests/
black src/ tests/
mypy src/
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes with tests
4. Run lint/typecheck: `ruff check && mypy src/`
5. Submit PR

---

## License

MIT License — see LICENSE file for details.

---

## Links

- **Phase 1 Details**: [phase1-rag-poc/README.md](phase1-rag-poc/README.md)
- **Phase 2 Details**: [phase2-ado-rag/README.md](phase2-ado-rag/README.md)
- **LLM Setup Guide**: [phase1-rag-poc/INSTRUCTIONS.md](phase1-rag-poc/INSTRUCTIONS.md)
- **OpenRouter Models**: https://openrouter.ai/models
- **Ollama Models**: https://ollama.com/library