"""
Central configuration for the RAG POC.

Reads from .env file and exposes typed settings to all modules.
Single source of truth — no other module should hardcode config values.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama (must be running locally) ---
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"

    # --- Embeddings (local) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Vector Store ---
    chroma_persist_dir: str = "./storage"
    chroma_collection_name: str = "rag_poc_v1"

    # --- Content-type-specific chunk sizes (tokens) ---
    chunk_size_pdf: int = 450
    chunk_overlap_pdf: int = 80
    chunk_size_url: int = 350
    chunk_overlap_url: int = 60
    chunk_size_chat: int = 200
    chunk_overlap_chat: int = 30

    # --- Retrieval ---
    top_k: int = 5
    relevance_threshold: float = 0.35

    # --- Paths ---
    synthetic_data_dir: str = "./data/synthetic"
    golden_qa_path: str = "./data/golden_qa/golden_qa_set.jsonl"
    eval_results_dir: str = "./data/eval_results"

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def synthetic_data_path(self) -> Path:
        return Path(self.synthetic_data_dir)

    @property
    def eval_results_path(self) -> Path:
        return Path(self.eval_results_dir)


# Singleton instance — import this everywhere
settings = Settings()
