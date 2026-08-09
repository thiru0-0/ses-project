"""
RAG pipeline orchestrator — adapted Corrective RAG pattern.

Phase 1 flow:
    query → retrieve → grade → (generate with citations OR decline)

Phase 2 additions:
    - HyDE query expansion (handled by the Retriever)
    - Test-case-shaped output mode for ADO user stories (8-field format)
    - Mode auto-detection or explicit selection
    - Human-review flag when confidence is below threshold
    - Session-scoped retrieval via metadata filtering

Simple, auditable. No self-critique loops, no multi-hop reasoning.
"""

import logging
from dataclasses import dataclass, field

from src.ragpoc.retrieval.vector_store import VectorStore
from src.ragpoc.retrieval.retriever import Retriever
from src.ragpoc.retrieval.relevance_grader import RelevanceGrader
from src.ragpoc.generation.prompt_templates import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEST_CASE,
    DECLINE_RESPONSE,
    LOW_CONFIDENCE_FLAG,
    format_qa_prompt,
    format_test_case_prompt,
    format_context_from_chunks,
)
from src.ragpoc.models.registry import get_llm_provider
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    """A source cited in the generated answer."""

    title: str
    source_ref: str
    chunk_id: str


@dataclass
class QueryResponse:
    """Response from the RAG pipeline."""

    answer: str  # generated answer or decline message
    sources: list[SourceCitation] = field(default_factory=list)
    confidence: float = 0.0  # grader's confidence score
    declined: bool = False  # True if grader declined
    retrieved_chunks: int = 0  # how many chunks were retrieved
    relevant_chunks: int = 0  # how many passed the threshold
    mode: str = "qa"  # "qa" or "test_case"
    needs_review: bool = False  # True when confidence < human_review_threshold


class RAGPipeline:
    """Orchestrates the retrieve → grade → generate/decline flow.

    Supports two output modes:
    - ``"qa"`` — free-text answer with citations (Phase 1 default).
    - ``"test_case"`` — structured 8-field test-case output (Phase 2).

    Mode can be set explicitly or auto-detected from the source type
    of the top retrieved chunk. Includes human-review flagging when
    confidence falls below the configured threshold.
    """

    def __init__(self, vector_store: VectorStore):
        self._retriever = Retriever(vector_store)
        self._grader = RelevanceGrader()
        self._llm = get_llm_provider()
        logger.info("RAGPipeline initialized")

    def query(
        self, question: str, mode: str | None = None, session_id: str = ""
    ) -> QueryResponse:
        """Process a user question through the full RAG pipeline.

        Steps:
            1. Retrieve top-k chunks (with optional HyDE expansion)
            2. Grade chunks for relevance
            3. If relevant chunks exist: generate grounded answer with citations
            4. If no relevant chunks: return static decline response

        Args:
            question: The user's natural-language question or user story.
            mode: Output mode — "qa" for free text, "test_case" for
                  structured test cases. If None, auto-detects from
                  retrieved chunk source types.
            session_id: Optional session ID for scoped retrieval.

        Returns:
            QueryResponse with answer, sources, confidence, and review flag.
        """
        logger.info("Pipeline processing query: '%s'", question[:100])

        # Step 1: Retrieve (HyDE expansion happens inside the Retriever)
        retrieved = self._retriever.retrieve(question, session_id=session_id)

        # Step 2: Grade
        grading = self._grader.grade(question, retrieved)

        # Step 3: Generate or Decline
        if grading.should_decline:
            logger.info("Pipeline DECLINING — no relevant chunks")
            return QueryResponse(
                answer=DECLINE_RESPONSE,
                sources=[],
                confidence=grading.confidence_score,
                declined=True,
                retrieved_chunks=grading.total_retrieved,
                relevant_chunks=grading.total_relevant,
                mode=mode or "qa",
                needs_review=False,
            )

        # Auto-detect mode if not specified
        if mode is None:
            mode = self._detect_mode(grading.relevant_chunks)

        # Build context from relevant chunks
        context = format_context_from_chunks(grading.relevant_chunks)

        # Select prompt and system prompt based on mode
        if mode == "test_case":
            prompt = format_test_case_prompt(context, question)
            system_prompt = SYSTEM_PROMPT_TEST_CASE
        else:
            prompt = format_qa_prompt(context, question)
            system_prompt = SYSTEM_PROMPT

        # Generate answer
        logger.info(
            "Generating %s answer from %d relevant chunks (confidence: %.4f)",
            mode,
            len(grading.relevant_chunks),
            grading.confidence_score,
        )
        answer = self._llm.generate(prompt, system_prompt=system_prompt)

        # Human-review flag: if confidence is below threshold, flag it
        needs_review = grading.confidence_score < settings.human_review_threshold
        if needs_review:
            logger.info(
                "Confidence %.4f < threshold %.4f — flagging for human review",
                grading.confidence_score,
                settings.human_review_threshold,
            )
            answer += LOW_CONFIDENCE_FLAG

        # Build source citations
        sources = []
        seen = set()
        for chunk in grading.relevant_chunks:
            key = (chunk.title, chunk.source_ref)
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceCitation(
                        title=chunk.title,
                        source_ref=chunk.source_ref,
                        chunk_id=chunk.chunk_id,
                    )
                )

        logger.info(
            "Pipeline produced %s answer with %d source citations (review=%s)",
            mode,
            len(sources),
            needs_review,
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=grading.confidence_score,
            declined=False,
            retrieved_chunks=grading.total_retrieved,
            relevant_chunks=len(grading.relevant_chunks),
            mode=mode,
            needs_review=needs_review,
        )

    @staticmethod
    def _detect_mode(chunks) -> str:
        """Auto-detect output mode from retrieved chunk source types.

        If any chunk is from an ADO work item or wiki, use test_case mode.
        Otherwise fall back to general QA.
        """
        ado_types = {"ado_work_item", "ado_wiki"}
        for chunk in chunks:
            src = getattr(chunk, "source_type", "")
            if src in ado_types:
                return "test_case"
        return "qa"
