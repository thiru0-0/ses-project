"""
Provider registry — the single swap point for the entire project.

Factory functions that read from config and return concrete providers.
Set LLM_PROVIDER in .env to switch:
  - "groq"   → Groq cloud (default)
  - "ollama" → local Ollama
"""

import logging
from functools import lru_cache

from src.ragpoc.models.base import EmbeddingProvider, LLMProvider
from src.ragpoc.models.embeddings import SentenceTransformerEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (singleton).

    Currently uses sentence-transformers with bge-small-en-v1.5.
    """
    logger.info("Initializing embedding provider: %s", settings.embedding_model)
    return SentenceTransformerEmbeddings(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider (singleton).

    Reads settings.llm_provider to decide which backend to use.
    """
    provider = settings.llm_provider.lower()
    logger.info("Initializing LLM provider: %s", provider)

    if provider == "groq":
        from src.ragpoc.models.groq_llm import GroqLLM
        return GroqLLM(model=settings.groq_model)
    elif provider == "ollama":
        from src.ragpoc.models.llm import OllamaLLM
        return OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: 'groq', 'ollama'. Set LLM_PROVIDER in .env."
        )


def get_llamaindex_llm():
    """Return a LlamaIndex-compatible Ollama LLM instance.

    Used by LlamaIndex's VectorStoreIndex and query engine.
    """
    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        request_timeout=120.0,
    )


def get_llamaindex_embed_model():
    """Return a LlamaIndex-compatible HuggingFace embedding model.

    Used by LlamaIndex's VectorStoreIndex for automatic embedding.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=settings.embedding_model)


def check_providers_health() -> dict[str, bool]:
    """Check health of all configured providers.

    Returns:
        Dictionary mapping provider name to availability status.
    """
    llm = get_llm_provider()
    provider = settings.llm_provider.lower()
    result = {
        "llm_available": llm.is_available(),
        "llm_provider": provider,
        "embedding_model": settings.embedding_model,
    }
    if provider == "ollama":
        result["llm_model"] = settings.llm_model
        result["ollama_url"] = settings.ollama_base_url
    elif provider == "groq":
        result["llm_model"] = settings.groq_model
    return result
