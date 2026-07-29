# Phase 2 ADO RAG - LLM Setup Instructions

This project extends Phase 1 RAG POC with Azure DevOps integration. The LLM setup is identical to Phase 1.

## Quick Start

**Option A: Local LLM (Ollama) — Recommended for development**
```bash
# 1. Install Ollama from https://ollama.com
# 2. Pull the model
ollama pull qwen2.5:14b

# 3. Start Ollama server
ollama serve

# 4. Configure .env (copy from .env.example)
cp .env.example .env
# Edit .env: LLM_PROVIDER=ollama, LLM_MODEL=qwen2.5:14b
```

**Option B: Cloud LLM (OpenRouter) — No local GPU needed**
```bash
# 1. Get API key at https://openrouter.ai/keys
# 2. Configure .env
cp .env.example .env
# Edit .env:
#   LLM_PROVIDER=openrouter
#   OPENROUTER_API_KEY=sk-or-...
#   LLM_MODEL=openai/gpt-4o-mini
```

---

## Detailed Setup

See **Phase 1 INSTRUCTIONS.md** for complete details:
- [Phase 1 INSTRUCTIONS.md](../phase1-rag-poc/INSTRUCTIONS.md)

The configuration is identical. Both phases use the same LLM abstraction layer.

---

## Phase 2 Specific Configuration

Additional settings in `.env` for Azure DevOps integration:

```env
# Azure DevOps (required for ADO features)
ADO_ORG_URL=https://dev.azure.com/yourorg
ADO_PROJECT=YourProject
ADO_PAT=your-personal-access-token  # Keep secret!

# ADO-specific chunking
CHUNK_SIZE_ADO=400
CHUNK_OVERLAP_ADO=70
```

### ADO Personal Access Token
1. Go to Azure DevOps → User Settings → Personal Access Tokens
2. Create new token with scopes:
   - **Work Items** (Read)
   - **Code** (Read) — for PR/repo content
   - **Wiki** (Read) — if using wiki pages
3. Copy token immediately (shown only once)
4. Add to `.env` as `ADO_PAT`

---

## Running Phase 2

```bash
cd phase2-ado-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate data & build index (includes ADO synthetic data)
python scripts/generate_synthetic_data.py
python scripts/build_index.py

# Start API
uvicorn src.ragpoc.api.main:app --reload --port 8001

# Start UI (separate terminal)
streamlit run ui/app.py --server.port 8502

# Run ADO evaluation
python scripts/run_ado_evaluation.py
```

---

## Key Differences from Phase 1

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| **Data Sources** | Synthetic docs, URLs, chats | + Azure DevOps work items, PRs, wiki |
| **Retrieval** | Vector similarity | + HyDE (Hypothetical Document Embeddings) |
| **Evaluation** | RAGAS on golden QA | + ADO-specific golden set |
| **API Port** | 8000 | 8001 |
| **UI Port** | 8501 | 8502 |

---

## Troubleshooting

See Phase 1 INSTRUCTIONS.md for:
- Ollama connection issues
- OpenRouter API key problems
- Model selection guidance
- Cost estimation

---

## Next Steps

- Run evaluation: `python -m src.ragpoc.evaluation.ragas_runner`
- Explore ADO ingestion: `python scripts/ado_ingestion_compare.py`
- Check API docs at http://localhost:8001/docs