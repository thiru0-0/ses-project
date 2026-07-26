"""
ChromaDB vector store wrapper integrated with LlamaIndex.

Uses ChromaDB's PersistentClient for on-disk storage and LlamaIndex's
ChromaVectorStore for seamless integration with the indexing pipeline.
"""

import logging
from pathlib import Path

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

    def __init__(self):
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

        logger.info(
            "VectorStore initialized: collection='%s', persist='%s', docs=%d",
            settings.chroma_collection_name,
            persist_path,
            self._collection.count(),
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store.

        Converts Chunk dataclasses to LlamaIndex TextNode objects
        and indexes them through the VectorStoreIndex.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        logger.info("Adding %d chunks to vector store", len(chunks))

        # Convert chunks to LlamaIndex TextNode objects
        nodes = []
        for chunk in chunks:
            node = TextNode(
                text=chunk.content,
                id_=chunk.chunk_id,
                metadata={
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
                },
                excluded_embed_metadata_keys=[
                    "doc_id",
                    "chunk_index",
                    "source_type",
                    "source_ref",
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
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                embed_model=self._embed_model,
            )
        return self._index

    def query_raw(
        self, query_text: str, top_k: int | None = None
    ) -> dict:
        """Query ChromaDB directly (bypassing LlamaIndex).

        Useful for the relevance grader which needs raw distance scores.

        Args:
            query_text: The query string.
            top_k: Number of results to return (defaults to settings.top_k).

        Returns:
            ChromaDB query results dictionary.
        """
        k = top_k or settings.top_k
        from src.ragpoc.models.registry import get_embedding_provider

        embed_provider = get_embedding_provider()
        query_embedding = embed_provider.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_stats(self) -> dict:
        """Return collection statistics.

        Returns:
            Dictionary with document count and collection name.
        """
        count = self._collection.count()
        return {
            "collection_name": settings.chroma_collection_name,
            "total_chunks": count,
            "persist_dir": str(settings.chroma_persist_dir),
        }

    def clear(self) -> None:
        """Clear all documents from the collection."""
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
