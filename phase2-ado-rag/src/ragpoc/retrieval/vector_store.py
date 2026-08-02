"""
ChromaDB vector store wrapper integrated with LlamaIndex.

Uses ChromaDB's PersistentClient for on-disk storage and LlamaIndex's
ChromaVectorStore for seamless integration with the indexing pipeline.
Supports session-based isolation via metadata filtering.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.ragpoc.ingestion.chunker import Chunk
from src.ragpoc.models.registry import (
    get_llamaindex_embed_model,
    get_llamaindex_llm,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store with LlamaIndex integration."""

    def __init__(self, session_id: Optional[str] = None):
        # Ensure storage directory exists
        persist_path = Path(settings.chroma_persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self._chroma_client = chromadb.PersistentClient(
            path=str(persist_path)
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name
        )

        # Create LlamaIndex vector store wrapper
        self._vector_store = ChromaVectorStore(
            chroma_collection=self._collection
        )

        # Set up storage context
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

        # LlamaIndex components
        self._embed_model = get_llamaindex_embed_model()
        self._index = None
        self._session_id = session_id

        logger.info(
            "VectorStore initialized: collection='%s', persist='%s', docs=%d, session_id=%s",
            settings.chroma_collection_name,
            persist_path,
            self._collection.count(),
            session_id,
        )

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def set_session(self, session_id: Optional[str]) -> None:
        self._session_id = session_id
        self._index = None  # Reset index to rebuild with new filter
        logger.info("VectorStore session changed to: %s", session_id)

    def add_chunks(self, chunks: list[Chunk], session_id: Optional[str] = None) -> int:
        """Add chunks to the vector store.

        Converts Chunk dataclasses to LlamaIndex TextNode objects
        and indexes them through the VectorStoreIndex.

        Args:
            chunks: List of Chunk objects to index.
            session_id: Optional session ID to associate with chunks.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        session_id = session_id or self._session_id
        logger.info("Adding %d chunks to vector store (session_id=%s)", len(chunks), session_id)

        # Convert chunks to LlamaIndex TextNode objects
        nodes = []
        for chunk in chunks:
            metadata = {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "source_type": chunk.source_type,
                "source_ref": chunk.source_ref,
                "title": chunk.title,
                **{
                    k: v
                    for k, v in chunk.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            if session_id:
                metadata["session_id"] = session_id

            node = TextNode(
                text=chunk.content,
                id_=chunk.chunk_id,
                metadata=metadata,
                excluded_embed_metadata_keys=[
                    "doc_id",
                    "chunk_index",
                    "source_type",
                    "source_ref",
                    "session_id",
                ],
                excluded_llm_metadata_keys=[
                    "doc_id",
                    "chunk_index",
                ],
            )
            nodes.append(node)

        # Build or update the index
        if self._index is None:
            self._index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self._storage_context,
                embed_model=self._embed_model,
                show_progress=True,
            )
        else:
            self._index.insert_nodes(nodes)

        logger.info(
            "Added %d chunks. Total documents in collection: %d",
            len(chunks),
            self._collection.count(),
        )
        return len(chunks)

    def get_index(self) -> VectorStoreIndex:
        """Get or build the LlamaIndex VectorStoreIndex.

        If no index exists yet, creates one from the existing collection.

        Returns:
            The VectorStoreIndex instance.
        """
        if self._index is None:
            if self._session_id:
                # Use LlamaIndex's metadata filtering
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self._vector_store,
                    embed_model=self._embed_model,
                )
            else:
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self._vector_store,
                    embed_model=self._embed_model,
                )
        return self._index

    def query_raw(
        self, query_text: str, top_k: int | None = None, session_id: Optional[str] = None
    ) -> dict:
        """Query ChromaDB directly (bypassing LlamaIndex).

        Useful for the relevance grader which needs raw distance scores.

        Args:
            query_text: The query string.
            top_k: Number of results to return (defaults to settings.top_k).
            session_id: Optional session ID to filter results.

        Returns:
            ChromaDB query results dictionary.
        """
        k = top_k or settings.top_k
        from src.ragpoc.models.registry import get_embedding_provider

        embed_provider = get_embedding_provider()
        query_embedding = embed_provider.embed_query(query_text)

        session_id = session_id or self._session_id
        where_clause = {"session_id": session_id} if session_id else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
            where=where_clause,
        )
        return results

    def get_stats(self, session_id: Optional[str] = None) -> dict:
        """Return collection statistics.

        Args:
            session_id: Optional session ID to filter stats.

        Returns:
            Dictionary with document count and collection name.
        """
        session_id = session_id or self._session_id

        if session_id:
            try:
                results = self._collection.get(where={"session_id": session_id})
                count = len(results.get("ids", []))
            except Exception:
                count = 0
        else:
            count = self._collection.count()

        return {
            "collection_name": settings.chroma_collection_name,
            "total_chunks": count,
            "persist_dir": str(settings.chroma_persist_dir),
            "session_id": session_id,
        }

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear all documents from the collection or for a specific session.

        Args:
            session_id: If provided, only clears chunks for that session.
                        If None, clears the entire collection.
        """
        if session_id:
            logger.warning("Clearing documents for session: %s", session_id)
            self._collection.delete(where={"session_id": session_id})
            self._index = None
            logger.info("Session %s data cleared", session_id)
        else:
            logger.warning("Clearing all documents from collection")
            self._chroma_client.delete_collection(
                name=settings.chroma_collection_name
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=settings.chroma_collection_name
            )
            self._vector_store = ChromaVectorStore(
                chroma_collection=self._collection
            )
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store
            )
            self._index = None
            logger.info("Collection cleared")
