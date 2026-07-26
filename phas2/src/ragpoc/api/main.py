"""
FastAPI application entrypoint.

Sets up CORS middleware, lifespan handler for model initialization,
and includes all route modules.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ragpoc.retrieval.vector_store import VectorStore
from src.ragpoc.generation.pipeline import RAGPipeline
from src.ragpoc.models.registry import check_providers_health

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Global instances (initialized during lifespan)
vector_store: VectorStore | None = None
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — initialize models on startup."""
    global vector_store, rag_pipeline

    logger.info("Starting RAG POC API...")

    # Check provider health
    health = check_providers_health()
    logger.info("Provider health: %s", health)

    # Initialize vector store and pipeline
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(vector_store)

    logger.info("RAG POC API ready")
    yield

    # Cleanup
    logger.info("Shutting down RAG POC API")


app = FastAPI(
    title="RAG POC API",
    description=(
        "Retrieval-Augmented Generation proof-of-concept API. "
        "Upload documents, share links, paste messages, and ask questions."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — allow Streamlit and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routes
from src.ragpoc.api.routes_ingest import router as ingest_router
from src.ragpoc.api.routes_query import router as query_router
from src.ragpoc.api.routes_evaluate import router as evaluate_router

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, tags=["Query"])
app.include_router(evaluate_router, prefix="/evaluate", tags=["Evaluation"])


@app.get("/health")
async def health_check():
    """Overall API health check."""
    provider_health = check_providers_health()
    store_stats = vector_store.get_stats() if vector_store else {}
    return {
        "status": "healthy",
        "providers": provider_health,
        "vector_store": store_stats,
    }
