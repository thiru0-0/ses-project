"""
Content-type-aware text chunker.

Splits NormalizedDocuments into Chunks using recursive character splitting
with content-type-specific parameters:
  - PDF/DOCX:  450 tokens, 80 overlap (longer structured prose)
  - URLs:      350 tokens, 60 overlap (already clean from Trafilatura)
  - Chat:      200 tokens, 30 overlap (short conversational turns)
"""

import logging
import re
from dataclasses import dataclass, field

from src.ragpoc.ingestion.normalizer import NormalizedDocument
from config.settings import settings

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for English text
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A text chunk ready for embedding and indexing."""

    chunk_id: str  # "{doc_id}_{chunk_index}"
    content: str  # chunk text
    doc_id: str  # parent document ID
    chunk_index: int  # position within document
    source_type: str  # inherited from NormalizedDocument
    source_ref: str  # inherited — needed for citations
    title: str  # inherited — needed for citations
    metadata: dict = field(default_factory=dict)


def get_chunk_params(source_type: str) -> tuple[int, int]:
    """Return (chunk_size_chars, overlap_chars) for the given source type.

    Converts token-based settings to character counts using an
    approximate 4 chars/token ratio.
    """
    match source_type:
        case "pdf" | "docx":
            size = settings.chunk_size_pdf
            overlap = settings.chunk_overlap_pdf
        case "url":
            size = settings.chunk_size_url
            overlap = settings.chunk_overlap_url
        case "message":
            size = settings.chunk_size_chat
            overlap = settings.chunk_overlap_chat
        case "ado_work_item" | "ado_wiki":
            size = settings.chunk_size_ado
            overlap = settings.chunk_overlap_ado
        case _:
            # Fallback to PDF/DOCX settings
            size = settings.chunk_size_pdf
            overlap = settings.chunk_overlap_pdf

    return size * CHARS_PER_TOKEN, overlap * CHARS_PER_TOKEN


def chunk_document(doc: NormalizedDocument) -> list[Chunk]:
    """Split a NormalizedDocument into chunks.

    Uses recursive splitting: paragraphs → sentences → words,
    with content-type-specific size and overlap parameters.

    Args:
        doc: The normalized document to chunk.

    Returns:
        List of Chunk objects with inherited metadata.
    """
    chunk_size, chunk_overlap = get_chunk_params(doc.source_type)

    logger.info(
        "Chunking doc '%s' (type=%s, len=%d chars) with size=%d, overlap=%d",
        doc.title,
        doc.source_type,
        len(doc.content),
        chunk_size,
        chunk_overlap,
    )

    # Split text into chunks
    text_chunks = _recursive_split(doc.content, chunk_size, chunk_overlap)

    # Convert to Chunk dataclass instances
    chunks = []
    for i, text in enumerate(text_chunks):
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_{i:04d}",
            content=text,
            doc_id=doc.doc_id,
            chunk_index=i,
            source_type=doc.source_type,
            source_ref=doc.source_ref,
            title=doc.title,
            metadata={
                **doc.metadata,
                "chunk_index": i,
                "chunk_count": len(text_chunks),
                "chunk_char_count": len(text),
                "ingested_at": doc.ingested_at,
            },
        )
        chunks.append(chunk)

    logger.info(
        "Document '%s' split into %d chunks", doc.title, len(chunks)
    )
    return chunks


def _recursive_split(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Recursively split text using a hierarchy of separators.

    Split priority: paragraphs → sentences → words
    """
    # If text fits in one chunk, return as-is
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try separators in order of preference
    separators = [
        "\n\n",  # paragraph boundaries
        "\n",  # line breaks
        ". ",  # sentence boundaries
        "? ",  # question marks
        "! ",  # exclamation marks
        "; ",  # semicolons
        ", ",  # commas
        " ",  # words
    ]

    for separator in separators:
        if separator in text:
            splits = _split_by_separator(
                text, separator, chunk_size, chunk_overlap
            )
            if splits:
                return splits

    # Last resort: hard character split
    return _hard_split(text, chunk_size, chunk_overlap)


def _split_by_separator(
    text: str, separator: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split text by a separator and merge into chunks of appropriate size."""
    parts = text.split(separator)
    chunks = []
    current_chunk = ""

    for part in parts:
        candidate = (
            f"{current_chunk}{separator}{part}" if current_chunk else part
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Save current chunk if non-empty
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # If the part alone is already too large, recursively split it
            # first — BEFORE prepending overlap — to avoid infinite recursion
            if len(part) > chunk_size:
                sub_chunks = _recursive_split(part, chunk_size, chunk_overlap)
                if sub_chunks:
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = ""
                continue

            # Start new chunk: try to prepend overlap from previous chunk
            if chunk_overlap > 0 and current_chunk:
                overlap_text = current_chunk[-chunk_overlap:]
                candidate_with_overlap = f"{overlap_text}{separator}{part}"
                # Only use overlap if it still fits
                current_chunk = (
                    candidate_with_overlap
                    if len(candidate_with_overlap) <= chunk_size
                    else part
                )
            else:
                current_chunk = part

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _hard_split(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Hard split by character count as a last resort."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap if chunk_overlap > 0 else end
        # Prevent infinite loop
        if start >= end:
            break
    return chunks
