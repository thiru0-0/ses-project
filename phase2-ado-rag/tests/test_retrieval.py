"""
Unit tests for the retrieval layer.

Tests relevance grader logic (the hallucination guardrail).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.ragpoc.retrieval.retriever import RetrievedChunk
from src.ragpoc.retrieval.relevance_grader import RelevanceGrader, GradingResult


def _make_chunk(chunk_id: str = "c1", content: str = "test") -> RetrievedChunk:
    """Helper to create a RetrievedChunk for testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        source_type="pdf",
        source_ref="test.pdf",
        title="Test Doc",
        distance=0.5,
        metadata={},
    )


class TestRelevanceGrader:
    """Tests for the relevance grader (hallucination guardrail)."""

    @patch("src.ragpoc.retrieval.relevance_grader.get_embedding_provider")
    def test_no_chunks_should_decline(self, mock_provider):
        """Empty chunk list should trigger decline."""
        grader = RelevanceGrader(threshold=0.35)
        result = grader.grade("test query", [])

        assert result.should_decline is True
        assert result.relevant_chunks == []
        assert result.confidence_score == 0.0
        assert result.total_retrieved == 0

    @patch("src.ragpoc.retrieval.relevance_grader.get_embedding_provider")
    def test_high_similarity_passes(self, mock_provider):
        """Chunks with high similarity should pass the grader."""
        mock_embed = MagicMock()
        # Query embedding
        mock_embed.embed_query.return_value = [1.0, 0.0, 0.0]
        # Chunk embedding — same direction = high similarity
        mock_embed.embed.return_value = [[0.95, 0.05, 0.0]]
        mock_provider.return_value = mock_embed

        grader = RelevanceGrader(threshold=0.35)
        chunks = [_make_chunk("c1", "relevant content")]
        result = grader.grade("test query", chunks)

        assert result.should_decline is False
        assert len(result.relevant_chunks) == 1
        assert result.confidence_score > 0.35

    @patch("src.ragpoc.retrieval.relevance_grader.get_embedding_provider")
    def test_low_similarity_declines(self, mock_provider):
        """Chunks with low similarity should trigger decline."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [1.0, 0.0, 0.0]
        # Orthogonal direction = ~0 similarity
        mock_embed.embed.return_value = [[0.0, 1.0, 0.0]]
        mock_provider.return_value = mock_embed

        grader = RelevanceGrader(threshold=0.35)
        chunks = [_make_chunk("c1", "irrelevant content")]
        result = grader.grade("test query", chunks)

        assert result.should_decline is True
        assert len(result.relevant_chunks) == 0
        assert result.total_retrieved == 1
        assert result.total_relevant == 0

    @patch("src.ragpoc.retrieval.relevance_grader.get_embedding_provider")
    def test_mixed_chunks_partial_pass(self, mock_provider):
        """Mix of relevant and irrelevant chunks — only relevant ones pass."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [1.0, 0.0, 0.0]
        # First chunk relevant, second irrelevant
        mock_embed.embed.return_value = [
            [0.9, 0.1, 0.0],  # high similarity
            [0.0, 1.0, 0.0],  # low similarity
        ]
        mock_provider.return_value = mock_embed

        grader = RelevanceGrader(threshold=0.35)
        chunks = [
            _make_chunk("c1", "relevant"),
            _make_chunk("c2", "irrelevant"),
        ]
        result = grader.grade("test query", chunks)

        assert result.should_decline is False
        assert len(result.relevant_chunks) == 1
        assert result.relevant_chunks[0].chunk_id == "c1"
        assert result.total_retrieved == 2
        assert result.total_relevant == 1

    def test_cosine_similarity_computation(self):
        """Test the cosine similarity static method."""
        query = np.array([1.0, 0.0, 0.0])
        chunks = np.array([
            [1.0, 0.0, 0.0],  # identical = 1.0
            [0.0, 1.0, 0.0],  # orthogonal = 0.0
            [-1.0, 0.0, 0.0],  # opposite = -1.0
        ])

        sims = RelevanceGrader._cosine_similarity(query, chunks)

        assert abs(sims[0] - 1.0) < 1e-6
        assert abs(sims[1] - 0.0) < 1e-6
        assert abs(sims[2] - (-1.0)) < 1e-6
