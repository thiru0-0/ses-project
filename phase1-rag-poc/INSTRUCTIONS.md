# Phase 1 RAG POC - LLM Setup Instructions

This guide explains how to configure the LLM provider for the Phase 1 RAG POC. You have **two options**:

- **Option A: Ollama (Local)** — Run models locally on your machine (free, private, requires GPU/RAM)
- **Option B: OpenRouter (Cloud)** — Use 100+ models via API (pay-per-token, no local GPU needed)

You only need **one** of these configured.

---

## Option A: Ollama (Local LLM) — Recommended for Development

### Prerequisites
- **macOS/Linux/Windows** with 16GB+ RAM (32GB recommended for 14B models)
- **GPU** (Apple Silicon, NVIDIA CUDA, or AMD ROCm) for acceptable speed

### Installation

**macOS (Apple Silicon):**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download)

### Pull the Model
```bash
# Default model used in this project (Qwen2.5 14B Instruct - ~9GB)
ollama pull qwen2.5:14b

# Alternative smaller models if RAM is limited:
# ollama pull qwen2.5:7b     # ~4.7GB
# ollama pull llama3.1:8b    # ~4.9GB
# ollama pull phi3:14b       # ~8.2GB
```

### Start Ollama Server
```bash
# Runs on http://localhost:11434 by default
ollama serve
```

### Verify
```bash
ollama list
# Should show qwen2.5:14b (or your chosen model)
```

### Configure `.env`
```bash
cd phase1-rag-poc
cp .env.example .env
```

Edit `.env` and ensure:
```env
# LLM Provider: ollama (default) or openrouter
LLM_PROVIDER=ollama

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:14b

# Embeddings (always local)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

> **Note**: If you pulled a different model (e.g., `llama3.1:8b`), update `LLM_MODEL` in `.env` accordingly.

---

## Option B: OpenRouter (Cloud LLM) — No Local GPU Needed

### Prerequisites
- **OpenRouter account**: Sign up at [openrouter.ai](https://openrouter.ai)
- **API Key**: Get from [openrouter.ai/keys](https://openrouter.ai/keys)
- **Credits**: Add credits to your account (pay-as-you-go)

### Recommended Models

| Model | Context | Cost (per 1M tokens) | Best For |
|-------|---------|---------------------|----------|
| `openai/gpt-4o-mini` | 128K | $0.15/$0.60 | Best quality/price balance |
| `anthropic/claude-3.5-haiku` | 200K | $0.25/$1.25 | Strong reasoning |
| `meta-llama/llama-3.1-70b-instruct` | 128K | $0.59/$0.79 | Open-source, good quality |
| `google/gemini-flash-1.5` | 1M | $0.075/$0.30 | Large context, low cost |
| `qwen/qwen-2.5-72b-instruct` | 32K | $0.35/$0.40 | Strong coding, multilingual |

### Configure `.env`
```bash
cd phase1-rag-poc
cp .env.example .env
```

Edit `.env`:
```env
# LLM Provider: openrouter
LLM_PROVIDER=openrouter

# OpenRouter settings
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini  # or your preferred model

# Optional: Site info for OpenRouter rankings
OPENROUTER_SITE_URL=https://github.com/yourusername/ses-rag
OPENROUTER_APP_NAME=SES-RAG-Phase1

# Embeddings (always local)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Model Name Format
OpenRouter uses **`provider/model-name`** format. Examples:
- `openai/gpt-4o-mini`
- `anthropic/claude-3.5-haiku`
- `meta-llama/llama-3.1-70b-instruct`
- `google/gemini-flash-1.5`
- `qwen/qwen-2.5-72b-instruct`

Browse all models at [openrouter.ai/models](https://openrouter.ai/models)

---

## Switching Between Providers

Simply change `LLM_PROVIDER` in `.env` and update the corresponding settings:

```env
# For Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:14b

# For OpenRouter
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openai/gpt-4o-mini
```

**No code changes required** — the application reads the provider from config at startup.

---

## Embedding Model (Always Local)

Both providers use **local embeddings** via `sentence-transformers`:
- Default: `BAAI/bge-small-en-v1.5` (33M params, ~130MB, fast, good quality)
- Alternative: `BAAI/bge-base-en-v1.5` (110M params, better quality, slower)
- Alternative: `sentence-transformers/all-MiniLM-L6-v2` (22M params, fastest)

Configure in `.env`:
```env
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

The model downloads automatically on first run (~130MB).

---

## Running the Application

### 1. Setup Environment
```bash
cd phase1-rag-poc
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Data & Build Index
```bash
python scripts/generate_synthetic_data.py
python scripts/build_index.py
```

### 3. Start API Server
```bash
uvicorn src.ragpoc.api.main:app --reload --port 8000
```

### 4. Start Streamlit UI (separate terminal)
```bash
streamlit run ui/app.py
```

### 5. Open UI
Visit http://localhost:8501

---

## Troubleshooting

### Ollama Issues

**"Connection refused" / "Model not found"**
```bash
# Ensure Ollama is running
ollama serve

# Check model exists
ollama list

# Pull if missing
ollama pull qwen2.5:14b
```

**Out of Memory (OOM)**
- Use smaller model: `ollama pull qwen2.5:7b` and update `LLM_MODEL=qwen2.5:7b`
- Or switch to OpenRouter (Option B)

**Slow Generation**
- Ensure GPU acceleration: `ollama ps` should show GPU usage
- macOS: Metal auto-enabled on Apple Silicon
- Linux/NVIDIA: Install `nvidia-container-toolkit`

### OpenRouter Issues

**"Invalid API Key"**
- Verify key at https://openrouter.ai/keys
- Check `.env` has no extra spaces/quotes

**"Model not found"**
- Use exact model ID from openrouter.ai/models (e.g., `openai/gpt-4o-mini`, not `gpt-4o-mini`)

**Rate Limits / Quota Exceeded**
- Add credits at openrouter.ai/credits
- Check usage at openrouter.ai/activity

**High Latency**
- Choose lower-latency model (e.g., `gpt-4o-mini` over `gpt-4o`)
- Consider geographic proximity to OpenRouter edge

---

## Cost Estimation (OpenRouter)

| Model | Input (1M tokens) | Output (1M tokens) | Est. Cost/Query* |
|-------|-------------------|-------------------|------------------|
| `gpt-4o-mini` | $0.15 | $0.60 | ~$0.001 |
| `claude-3.5-haiku` | $0.25 | $1.25 | ~$0.002 |
| `llama-3.1-70b` | $0.59 | $0.79 | ~$0.002 |
| `gemini-flash-1.5` | $0.075 | $0.30 | ~$0.0005 |

*Assuming ~2K input + 500 output tokens per query

---

## Next Steps

- Read `README.md` for project overview
- Run evaluation: `python -m src.ragpoc.evaluation.ragas_runner`
- Proceed to **Phase 2** for Azure DevOps integration: `../phase2-ado-rag/INSTRUCTIONS.md`