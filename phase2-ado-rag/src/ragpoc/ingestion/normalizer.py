"""
Document normalizer — the convergence point.

Takes any RawDocument (from PDF, DOCX, URL, or pasted message) and
produces a NormalizedDocument with a consistent schema. This is the
contract between ingestion (Person A) and retrieval (Person B).

Person B's code only ever sees NormalizedDocument, never raw PDFs or HTML.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.ragpoc.ingestion.loaders import RawDocument

logger = logging.getLogger(__name__)


@dataclass
class NormalizedDocument:
    """Normalized document ready for chunking.

    This is the contract between ingestion and retrieval.
    All source types converge into this single schema.
    """

    doc_id: str  # UUID
    content: str  # cleaned text (whitespace-normalized, encoding-fixed)
    source_type: str  # "pdf" | "docx" | "url" | "message"
    source_ref: str  # original path/URL
    title: str  # extracted or inferred title
    ingested_at: str  # ISO 8601 timestamp
    metadata: dict = field(default_factory=dict)


def normalize(raw_doc: RawDocument, session_id: str = "") -> NormalizedDocument:
    """Normalize a RawDocument into a NormalizedDocument.

    Applies the following normalizations:
    - Strips excessive whitespace and blank lines
    - Fixes common encoding artifacts
    - Extracts or infers a title
    - Assigns a UUID and ingestion timestamp

    Args:
        raw_doc: The raw document from any loader.

    Returns:
        NormalizedDocument with cleaned, consistent content.

    Raises:
        ValueError: If the document has no usable content after cleaning.
    """
    logger.info(
        "Normalizing %s document from '%s'",
        raw_doc.source_type,
        raw_doc.source_ref,
    )

    # Clean the content
    content = _clean_text(raw_doc.content)

    if not content.strip():
        raise ValueError(
            f"Document from '{raw_doc.source_ref}' has no content after normalization"
        )

    # Extract or infer title
    title = _extract_title(raw_doc, content)

    # Build metadata (merge raw doc metadata with normalized metadata)
    metadata = {**raw_doc.metadata}
    metadata["original_char_count"] = len(raw_doc.content)
    metadata["normalized_char_count"] = len(content)
    if session_id:
        metadata["session_id"] = session_id

    return NormalizedDocument(
        doc_id=str(uuid.uuid4()),
        content=content,
        source_type=raw_doc.source_type,
        source_ref=raw_doc.source_ref,
        title=title,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        metadata=metadata,
    )


def _clean_text(text: str) -> str:
    """Clean and normalize text content.

    - Fix common encoding artifacts
    - Normalize whitespace (collapse multiple spaces/newlines)
    - Strip leading/trailing whitespace
    """
    # Fix common encoding artifacts
    replacements = {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",  # non-breaking space
        "\u200b": "",  # zero-width space
        "\ufeff": "",  # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse multiple blank lines into max 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into single space (per line)
    lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)
    text = "\n".join(lines)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def _extract_title(raw_doc: RawDocument, content: str) -> str:
    """Extract or infer a title from the document.

    Priority:
    1. Title from metadata (e.g., URL title from Trafilatura)
    2. First markdown heading in content
    3. Filename (for file-based sources)
    4. First non-empty line (truncated)
    5. Fallback: source type + truncated source ref
    """
    # 1. Check metadata for title
    meta_title = raw_doc.metadata.get("title", "").strip()
    if meta_title:
        return meta_title

    # 2. Look for first markdown heading
    heading_match = re.search(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()

    # 3. Use filename for file-based sources
    if raw_doc.source_type in ("pdf", "docx"):
        path = Path(raw_doc.source_ref)
        return path.stem.replace("_", " ").replace("-", " ").title()

    # 4. First non-empty line (truncated to 100 chars)
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:100]

    # 5. Fallback
    return f"{raw_doc.source_type}: {raw_doc.source_ref[:50]}"
