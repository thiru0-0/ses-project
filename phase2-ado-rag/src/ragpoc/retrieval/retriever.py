"""
Retriever — thin orchestration layer over the vector store.

Provides a clean interface for the generation pipeline to retrieve
relevant chunks. Supports optional HyDE (Hypothetical Document
Embeddings) query expansion: when enabled, the raw query is first
transformed into a hypothetical answer by the LLM, and that answer
is embedded for retrieval instead of the raw query text.

Separate from vector_store.py so we can add hybrid search or
reranking later without touching the pipeline.
"""

import logging
import time
from dataclasses import dataclass

from src.ragpoc.retrieval.vector_store import VectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its relevance score."""

    chunk_id: str
    content: str
    source_type: str
    source_ref: str
    title: str
    distance: float  # lower = more similar for L2, higher for cosine
    metadata: dict


class Retriever:
    """Retrieves relevant chunks from the vector store.

    Supports two retrieval modes:
    - **Direct**: embed the raw query and search (Phase 1 default).
    - **HyDE**: first generate a hypothetical answer document via the
      LLM, embed *that*, then search. This closes the wording gap
      between how users phrase questions and how source documents
      (especially test cases) are written.

    HyDE is toggled via ``settings.hyde_enabled``.
    """

    def __init__(self, vector_store: VectorStore, use_hyde: bool | None = None):
        self._vector_store = vector_store
        self._use_hyde = use_hyde if use_hyde is not None else settings.hyde_enabled

        if self._use_hyde:
            logger.info("Retriever initialized with HyDE ENABLED")
        else:
            logger.info("Retriever initialized (direct embedding, no HyDE)")

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query.

        When HyDE is enabled, the query is first expanded into a
        hypothetical document before being embedded for retrieval.

        Args:
            query: The user's question.
            top_k: Number of chunks to retrieve (defaults to settings.top_k).

        Returns:
            List of RetrievedChunk objects, ranked by relevance.
        """
        k = top_k or settings.top_k

        # Determine the text to embed for retrieval
        retrieval_text = query
        if self._use_hyde:
            retrieval_text = self._expand_with_hyde(query)

        logger.info(
            "Retrieving top-%d chunks for %squery: '%s'",
            k,
            "HyDE-expanded " if self._use_hyde else "",
            retrieval_text[:120],
        )

        results = self._vector_store.query_raw(retrieval_text, top_k=k)

        chunks = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

            for chunk_id, doc, meta, dist in zip(
                ids, documents, metadatas, distances
            ):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        content=doc,
                        source_type=meta.get("source_type", "unknown"),
                        source_ref=meta.get("source_ref", "unknown"),
                        title=meta.get("title", "Untitled"),
                        distance=dist,
                        metadata=meta,
                    )
                )

        logger.info("Retrieved %d chunks", len(chunks))
        return chunks

    def _expand_with_hyde(self, query: str) -> str:
        """Generate a hypothetical document via the LLM for HyDE retrieval.

        Falls back to the raw query if HyDE generation fails, so the
        pipeline degrades gracefully rather than erroring.
        """
        from src.ragpoc.models.registry import get_llm_provider

        llm = get_llm_provider()
        t0 = time.perf_counter()

        try:
            hyde_doc = llm.generate_hyde_document(query)
            elapsed = time.perf_counter() - t0
            logger.info(
                "HyDE expansion completed in %.2fs — using hypothetical "
                "document (%d chars) for retrieval",
                elapsed,
                len(hyde_doc),
            )
            return hyde_doc
        except Exception as e:
            logger.warning(
                "HyDE expansion failed (%s), falling back to raw query", e
            )
            return query
