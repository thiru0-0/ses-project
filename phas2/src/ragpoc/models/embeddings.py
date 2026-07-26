"""
Embedding provider using sentence-transformers (bge-small-en-v1.5).

Runs entirely on-device — no data leaves the machine.
Auto-detects MPS (Apple Silicon) / CUDA / CPU.
"""

import logging
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from src.ragpoc.models.base import EmbeddingProvider
from config.settings import settings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._device = self._detect_device()
        logger.info(
            "Loading embedding model '%s' on device '%s'",
            self._model_name,
            self._device,
        )
        self._model = SentenceTransformer(
            self._model_name, device=self._device
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded. Dimension: %d", self._dimension
        )

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embedding = self._model.encode(
            query, show_progress_bar=False, convert_to_numpy=True
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device


@lru_cache(maxsize=1)
def get_cached_embedding_provider() -> SentenceTransformerEmbeddings:
    """Return a singleton embedding provider instance."""
    return SentenceTransformerEmbeddings()
