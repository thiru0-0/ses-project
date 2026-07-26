"""
Relevance grader — the hallucination guardrail.

Scores retrieved chunks against the query and determines whether
the pipeline should generate an answer or decline. This is the most
stakeholder-visible piece of Phase 1, since "no hallucinated answers"
is a literal success criterion.

When no chunk clears the relevance threshold, the pipeline returns
a static decline message — no LLM call, no web fallback, no hallucination.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.ragpoc.retrieval.retriever import RetrievedChunk
from src.ragpoc.models.registry import get_embedding_provider
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """Result of relevance grading on retrieved chunks."""

    relevant_chunks: list[RetrievedChunk]  # chunks above threshold
    should_decline: bool  # True if no chunk is relevant enough
    confidence_score: float  # average relevance of passing chunks (0-1)
    total_retrieved: int  # how many chunks were initially retrieved
    total_relevant: int  # how many passed the threshold


class RelevanceGrader:
    """Grades retrieved chunks for relevance to prevent hallucination.

    Uses cosine similarity between query embedding and chunk embeddings
    to filter out irrelevant chunks. When no chunk passes the threshold,
    signals the pipeline to decline answering rather than hallucinate.

    When HyDE is enabled, the similarity distribution shifts because
    the retrieval query is a hypothetical document rather than the
    original user question. A separate ``hyde_threshold`` (defaulting
    to 0.30) accounts for this — tune it experimentally.
    """

    # Default threshold when HyDE is active. HyDE-expanded queries
    # tend to produce slightly lower cosine similarities because the
    # hypothetical document and the real document are both long-form
    # but worded differently. 0.30 is a starting point — retune after
    # running the ADO golden set.
    DEFAULT_HYDE_THRESHOLD = 0.30

    def __init__(
        self,
        threshold: float | None = None,
        hyde_threshold: float | None = None,
    ):
        self._base_threshold = threshold or settings.relevance_threshold
        self._hyde_threshold = hyde_threshold or self.DEFAULT_HYDE_THRESHOLD
        # Select active threshold based on config
        self._threshold = (
            self._hyde_threshold if settings.hyde_enabled else self._base_threshold
        )
        self._embed_provider = get_embedding_provider()
        logger.info(
            "RelevanceGrader initialized with threshold=%.2f "
            "(hyde_enabled=%s, base=%.2f, hyde=%.2f)",
            self._threshold,
            settings.hyde_enabled,
            self._base_threshold,
            self._hyde_threshold,
        )

    def grade(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> GradingResult:
        """Grade retrieved chunks for relevance to the query.

        Args:
            query: The user's question.
            chunks: Retrieved chunks to evaluate.

        Returns:
            GradingResult with filtered chunks and decline decision.
        """
        if not chunks:
            logger.info("No chunks to grade — declining")
            return GradingResult(
                relevant_chunks=[],
                should_decline=True,
                confidence_score=0.0,
                total_retrieved=0,
                total_relevant=0,
            )

        # Compute cosine similarities
        query_embedding = np.array(
            self._embed_provider.embed_query(query)
        )
        chunk_texts = [c.content for c in chunks]
        chunk_embeddings = np.array(
            self._embed_provider.embed(chunk_texts)
        )

        similarities = self._cosine_similarity(
            query_embedding, chunk_embeddings
        )

        # Filter by threshold
        relevant_chunks = []
        for chunk, sim in zip(chunks, similarities):
            logger.debug(
                "Chunk '%s' similarity: %.4f (threshold: %.4f)",
                chunk.chunk_id,
                sim,
                self._threshold,
            )
            if sim >= self._threshold:
                relevant_chunks.append(chunk)

        # Compute confidence
        if relevant_chunks:
            relevant_sims = [
                sim
                for sim in similarities
                if sim >= self._threshold
            ]
            confidence = float(np.mean(relevant_sims))
        else:
            confidence = 0.0

        should_decline = len(relevant_chunks) == 0

        if should_decline:
            logger.info(
                "Grading result: DECLINE — no chunks above threshold %.2f "
                "(best similarity: %.4f)",
                self._threshold,
                float(max(similarities)) if len(similarities) > 0 else 0.0,
            )
        else:
            logger.info(
                "Grading result: PASS — %d/%d chunks relevant "
                "(confidence: %.4f)",
                len(relevant_chunks),
                len(chunks),
                confidence,
            )

        return GradingResult(
            relevant_chunks=relevant_chunks,
            should_decline=should_decline,
            confidence_score=confidence,
            total_retrieved=len(chunks),
            total_relevant=len(relevant_chunks),
        )

    @staticmethod
    def _cosine_similarity(
        query_vec: np.ndarray, chunk_vecs: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and each chunk vector.

        Args:
            query_vec: 1D query embedding.
            chunk_vecs: 2D array of chunk embeddings (N x D).

        Returns:
            1D array of cosine similarities.
        """
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        chunk_norms = chunk_vecs / (
            np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-10
        )
        # Dot product = cosine similarity (since vectors are normalized)
        return chunk_norms @ query_norm
