"""
Unit tests for the ingestion pipeline.

Tests loaders, normalizer, and chunker.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ragpoc.ingestion.loaders import (
    RawDocument,
    load_message,
)
from src.ragpoc.ingestion.normalizer import (
    NormalizedDocument,
    normalize,
    _clean_text,
    _extract_title,
)
from src.ragpoc.ingestion.chunker import (
    Chunk,
    chunk_document,
    get_chunk_params,
    CHARS_PER_TOKEN,
)


# ═══════════════════════════════════════════════
# LOADER TESTS
# ═══════════════════════════════════════════════

class TestLoadMessage:
    """Tests for the pasted message loader."""

    def test_load_message_basic(self):
        raw = load_message("Hello, this is a test message.")
        assert raw.content == "Hello, this is a test message."
        assert raw.source_type == "message"
        assert raw.source_ref == "pasted"
        assert "pasted_at" in raw.metadata
        assert raw.metadata["char_count"] == 30

    def test_load_message_custom_label(self):
        raw = load_message("Test", source_label="slack-export")
        assert raw.source_ref == "slack-export"

    def test_load_message_strips_whitespace(self):
        raw = load_message("  hello world  \n  ")
        assert raw.content == "hello world"

    def test_load_message_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot load empty message"):
            load_message("")

    def test_load_message_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            load_message("   \n\n  ")

    def test_load_message_multiline(self):
        text = "Line 1\nLine 2\nLine 3"
        raw = load_message(text)
        assert raw.metadata["line_count"] == 3


# ═══════════════════════════════════════════════
# NORMALIZER TESTS
# ═══════════════════════════════════════════════

class TestCleanText:
    """Tests for text cleaning utilities."""

    def test_fixes_smart_quotes(self):
        text = "\u201cHello\u201d and \u2018world\u2019"
        assert _clean_text(text) == '"Hello" and \'world\''

    def test_fixes_dashes(self):
        text = "value\u2013range\u2014end"
        assert _clean_text(text) == "value-range-end"

    def test_collapses_blank_lines(self):
        text = "a\n\n\n\n\nb"
        assert _clean_text(text) == "a\n\nb"

    def test_collapses_spaces(self):
        text = "hello    world   test"
        assert _clean_text(text) == "hello world test"

    def test_strips_zero_width_chars(self):
        text = "hello\u200b\ufeffworld"
        assert _clean_text(text) == "helloworld"


class TestNormalize:
    """Tests for document normalization."""

    def test_normalize_produces_valid_doc(self):
        raw = RawDocument(
            content="# Test Document\n\nSome content here.",
            source_type="message",
            source_ref="pasted",
            metadata={"key": "value"},
        )
        doc = normalize(raw)

        assert isinstance(doc, NormalizedDocument)
        assert doc.doc_id  # UUID should be non-empty
        assert doc.content == "# Test Document\n\nSome content here."
        assert doc.source_type == "message"
        assert doc.source_ref == "pasted"
        assert doc.title == "Test Document"  # extracted from heading
        assert doc.ingested_at  # should have timestamp
        assert doc.metadata["key"] == "value"
        assert "original_char_count" in doc.metadata
        assert "normalized_char_count" in doc.metadata

    def test_normalize_empty_raises(self):
        raw = RawDocument(
            content="   ",
            source_type="message",
            source_ref="pasted",
            metadata={},
        )
        with pytest.raises(ValueError, match="no content"):
            normalize(raw)

    def test_normalize_title_from_filename(self):
        raw = RawDocument(
            content="Some content without a heading.",
            source_type="pdf",
            source_ref="/path/to/my_document.pdf",
            metadata={},
        )
        doc = normalize(raw)
        assert doc.title == "My Document"

    def test_normalize_title_from_metadata(self):
        raw = RawDocument(
            content="Some content.",
            source_type="url",
            source_ref="https://example.com",
            metadata={"title": "My Article Title"},
        )
        doc = normalize(raw)
        assert doc.title == "My Article Title"


# ═══════════════════════════════════════════════
# CHUNKER TESTS
# ═══════════════════════════════════════════════

class TestGetChunkParams:
    """Tests for content-type-specific chunk parameters."""

    def test_pdf_params(self):
        size, overlap = get_chunk_params("pdf")
        assert size == 450 * CHARS_PER_TOKEN
        assert overlap == 80 * CHARS_PER_TOKEN

    def test_docx_params(self):
        size, overlap = get_chunk_params("docx")
        assert size == 450 * CHARS_PER_TOKEN
        assert overlap == 80 * CHARS_PER_TOKEN

    def test_url_params(self):
        size, overlap = get_chunk_params("url")
        assert size == 350 * CHARS_PER_TOKEN
        assert overlap == 60 * CHARS_PER_TOKEN

    def test_message_params(self):
        size, overlap = get_chunk_params("message")
        assert size == 200 * CHARS_PER_TOKEN
        assert overlap == 30 * CHARS_PER_TOKEN

    def test_unknown_falls_back_to_pdf(self):
        size, overlap = get_chunk_params("unknown")
        assert size == 450 * CHARS_PER_TOKEN


class TestChunkDocument:
    """Tests for document chunking."""

    def test_short_doc_single_chunk(self):
        doc = NormalizedDocument(
            doc_id="test-id",
            content="Short document content.",
            source_type="message",
            source_ref="pasted",
            title="Test",
            ingested_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short document content."
        assert chunks[0].doc_id == "test-id"
        assert chunks[0].chunk_index == 0
        assert chunks[0].source_type == "message"
        assert chunks[0].title == "Test"

    def test_long_doc_multiple_chunks(self):
        # Create a document longer than chunk size
        content = "\n\n".join([f"Paragraph {i}. " * 50 for i in range(20)])
        doc = NormalizedDocument(
            doc_id="test-id",
            content=content,
            source_type="pdf",
            source_ref="test.pdf",
            title="Test",
            ingested_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        chunks = chunk_document(doc)
        assert len(chunks) > 1

        # Verify chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"test-id_{i:04d}"
            assert chunk.chunk_index == i

    def test_chunk_metadata_inheritance(self):
        doc = NormalizedDocument(
            doc_id="test-id",
            content="Test content for chunking.",
            source_type="url",
            source_ref="https://example.com",
            title="Example",
            ingested_at="2024-01-01T00:00:00Z",
            metadata={"custom_key": "custom_value"},
        )
        chunks = chunk_document(doc)
        assert chunks[0].source_type == "url"
        assert chunks[0].source_ref == "https://example.com"
        assert chunks[0].title == "Example"
        assert chunks[0].metadata["custom_key"] == "custom_value"
        assert chunks[0].metadata["ingested_at"] == "2024-01-01T00:00:00Z"
