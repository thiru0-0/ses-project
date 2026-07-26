"""
Unit tests for the generation pipeline.

Tests prompt formatting, decline behavior, and pipeline orchestration.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.ragpoc.generation.prompt_templates import (
    SYSTEM_PROMPT,
    DECLINE_RESPONSE,
    format_qa_prompt,
    format_context_from_chunks,
)
from src.ragpoc.generation.pipeline import (
    RAGPipeline,
    QueryResponse,
    SourceCitation,
)
from src.ragpoc.retrieval.retriever import RetrievedChunk
from src.ragpoc.retrieval.relevance_grader import GradingResult


# ═══════════════════════════════════════════════
# PROMPT TEMPLATE TESTS
# ═══════════════════════════════════════════════

class TestPromptTemplates:
    """Tests for prompt template formatting."""

    def test_system_prompt_contains_key_instructions(self):
        assert "retrieval assistant" in SYSTEM_PROMPT.lower()
        assert "context" in SYSTEM_PROMPT.lower()
        assert "prior knowledge" in SYSTEM_PROMPT.lower()

    def test_decline_response_is_informative(self):
        assert "couldn't find" in DECLINE_RESPONSE.lower()
        assert "provided sources" in DECLINE_RESPONSE.lower()

    def test_format_qa_prompt(self):
        prompt = format_qa_prompt(
            context="Some relevant information.",
            question="What is the answer?",
        )
        assert "Some relevant information." in prompt
        assert "What is the answer?" in prompt
        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt

    def test_format_context_from_chunks(self):
        chunks = [
            MagicMock(
                title="Doc A",
                source_ref="doc_a.pdf",
                content="Content from doc A.",
            ),
            MagicMock(
                title="Doc B",
                source_ref="https://example.com",
                content="Content from doc B.",
            ),
        ]
        context = format_context_from_chunks(chunks)

        assert "[Source: Doc A]" in context
        assert "[Source: Doc B]" in context
        assert "Content from doc A." in context
        assert "Content from doc B." in context
        assert "doc_a.pdf" in context


# ═══════════════════════════════════════════════
# PIPELINE TESTS
# ═══════════════════════════════════════════════

class TestRAGPipeline:
    """Tests for the RAG pipeline orchestrator."""

    def _make_chunk(self, chunk_id="c1", content="test", title="Doc"):
        return RetrievedChunk(
            chunk_id=chunk_id,
            content=content,
            source_type="pdf",
            source_ref="test.pdf",
            title=title,
            distance=0.5,
            metadata={},
        )

    @patch("src.ragpoc.generation.pipeline.get_llm_provider")
    @patch("src.ragpoc.generation.pipeline.RelevanceGrader")
    @patch("src.ragpoc.generation.pipeline.Retriever")
    def test_decline_when_no_relevant_chunks(
        self, MockRetriever, MockGrader, mock_llm_factory
    ):
        """Pipeline should return decline response when grader declines."""
        # Set up mocks
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        MockRetriever.return_value = mock_retriever

        mock_grader = MagicMock()
        mock_grader.grade.return_value = GradingResult(
            relevant_chunks=[],
            should_decline=True,
            confidence_score=0.0,
            total_retrieved=0,
            total_relevant=0,
        )
        MockGrader.return_value = mock_grader

        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm

        # Create pipeline and query
        mock_store = MagicMock()
        pipeline = RAGPipeline(mock_store)
        result = pipeline.query("What is the meaning of life?")

        assert result.declined is True
        assert result.answer == DECLINE_RESPONSE
        assert result.sources == []
        assert result.confidence == 0.0
        # LLM should NOT have been called
        mock_llm.generate.assert_not_called()

    @patch("src.ragpoc.generation.pipeline.get_llm_provider")
    @patch("src.ragpoc.generation.pipeline.RelevanceGrader")
    @patch("src.ragpoc.generation.pipeline.Retriever")
    def test_generate_with_citations(
        self, MockRetriever, MockGrader, mock_llm_factory
    ):
        """Pipeline should generate answer with citations when chunks are relevant."""
        chunk = self._make_chunk("c1", "The memory limit is 4GB.", "Spec v1")

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [chunk]
        MockRetriever.return_value = mock_retriever

        mock_grader = MagicMock()
        mock_grader.grade.return_value = GradingResult(
            relevant_chunks=[chunk],
            should_decline=False,
            confidence_score=0.85,
            total_retrieved=1,
            total_relevant=1,
        )
        MockGrader.return_value = mock_grader

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "The memory limit is 4GB [Source: Spec v1]."
        )
        mock_llm_factory.return_value = mock_llm

        mock_store = MagicMock()
        pipeline = RAGPipeline(mock_store)
        result = pipeline.query("What is the memory limit?")

        assert result.declined is False
        assert "4GB" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0].title == "Spec v1"
        assert result.confidence == 0.85
        # LLM should have been called with system prompt
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        assert call_args.kwargs.get("system_prompt") == SYSTEM_PROMPT

    @patch("src.ragpoc.generation.pipeline.get_llm_provider")
    @patch("src.ragpoc.generation.pipeline.RelevanceGrader")
    @patch("src.ragpoc.generation.pipeline.Retriever")
    def test_deduplicates_source_citations(
        self, MockRetriever, MockGrader, mock_llm_factory
    ):
        """Pipeline should deduplicate source citations."""
        chunk1 = self._make_chunk("c1", "Chunk 1 text.", "Same Doc")
        chunk2 = self._make_chunk("c2", "Chunk 2 text.", "Same Doc")

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [chunk1, chunk2]
        MockRetriever.return_value = mock_retriever

        mock_grader = MagicMock()
        mock_grader.grade.return_value = GradingResult(
            relevant_chunks=[chunk1, chunk2],
            should_decline=False,
            confidence_score=0.9,
            total_retrieved=2,
            total_relevant=2,
        )
        MockGrader.return_value = mock_grader

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Answer."
        mock_llm_factory.return_value = mock_llm

        mock_store = MagicMock()
        pipeline = RAGPipeline(mock_store)
        result = pipeline.query("test")

        # Should have only 1 unique source, not 2
        assert len(result.sources) == 1
        assert result.sources[0].title == "Same Doc"
