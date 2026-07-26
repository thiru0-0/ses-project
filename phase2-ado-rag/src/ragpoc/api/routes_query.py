"""
Query API routes.

POST /query     — ask a question, get a grounded answer
GET  /query/health — quick health check for query pipeline
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class SourceCitationResponse(BaseModel):
    title: str
    source_ref: str
    chunk_id: str


class QueryResponseModel(BaseModel):
    answer: str
    sources: list[SourceCitationResponse]
    confidence: float
    declined: bool
    retrieved_chunks: int
    relevant_chunks: int


def _get_pipeline():
    """Get the global RAG pipeline instance."""
    from src.ragpoc.api.main import rag_pipeline

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline


@router.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequest):
    """Ask a question and get a grounded answer.

    The question is processed through:
        retrieve → grade → generate (or decline if no relevant context)
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info("Query: '%s'", request.question[:100])

    try:
        pipeline = _get_pipeline()
        result = pipeline.query(request.question)

        return QueryResponseModel(
            answer=result.answer,
            sources=[
                SourceCitationResponse(
                    title=s.title,
                    source_ref=s.source_ref,
                    chunk_id=s.chunk_id,
                )
                for s in result.sources
            ],
            confidence=result.confidence,
            declined=result.declined,
            retrieved_chunks=result.retrieved_chunks,
            relevant_chunks=result.relevant_chunks,
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.get("/query/health")
async def query_health():
    """Quick health check for the query pipeline."""
    pipeline = _get_pipeline()
    return {"status": "ready", "pipeline": "initialized"}
