"""
Ingestion API routes.

POST /ingest/file      — multipart file upload (PDF, DOCX)
POST /ingest/url       — JSON body with URL to scrape
POST /ingest/message   — JSON body with pasted text
POST /ingest/auto      — unified smart ingest (auto-detects file/URL/text)
GET  /ingest/status    — returns collection stats
"""

import logging
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.ragpoc.ingestion.loaders import load_pdf, load_docx, load_url, load_message
from src.ragpoc.ingestion.normalizer import normalize
from src.ragpoc.ingestion.chunker import chunk_document

logger = logging.getLogger(__name__)
router = APIRouter()

# Regex for URL detection in pasted text
URL_REGEX = re.compile(
    r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE
)


class URLRequest(BaseModel):
    url: str
    session_id: str = ""


class MessageRequest(BaseModel):
    text: str
    source_label: str = "pasted"
    session_id: str = ""


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    chunk_count: int
    message: str


def _get_vector_store():
    """Get the global vector store instance."""
    from src.ragpoc.api.main import vector_store

    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store


def _ingest_document(raw_doc, session_id: str = ""):
    """Common ingestion flow: normalize → chunk → add to vector store."""
    store = _get_vector_store()
    normalized = normalize(raw_doc, session_id=session_id)
    chunks = chunk_document(normalized)
    store.add_chunks(chunks)
    return IngestResponse(
        doc_id=normalized.doc_id,
        title=normalized.title,
        source_type=normalized.source_type,
        chunk_count=len(chunks),
        message=f"Successfully ingested '{normalized.title}' ({len(chunks)} chunks)",
    )


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    session_id: str = Form(""),
):
    """Upload a PDF or DOCX file for ingestion.

    The file is processed through: load → normalize → chunk → index.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use .pdf or .docx",
        )

    logger.info("Ingesting file: %s (session=%s)", file.filename, session_id or "global")

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load based on file type
        if suffix == ".pdf":
            raw_doc = load_pdf(tmp_path)
        else:
            raw_doc = load_docx(tmp_path)

        # Override source_ref with original filename
        raw_doc.source_ref = file.filename
        raw_doc.metadata["filename"] = file.filename

        return _ingest_document(raw_doc, session_id=session_id)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("File ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: URLRequest):
    """Scrape and ingest content from a URL.

    Uses Trafilatura for boilerplate removal — raw HTML is never chunked.
    """
    logger.info("Ingesting URL: %s (session=%s)", request.url, request.session_id or "global")

    try:
        raw_doc = load_url(request.url)
        return _ingest_document(raw_doc, session_id=request.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("URL ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.post("/message", response_model=IngestResponse)
async def ingest_message(request: MessageRequest):
    """Ingest a pasted message or chat thread."""
    logger.info(
        "Ingesting pasted message (%d chars, session=%s)",
        len(request.text),
        request.session_id or "global",
    )

    try:
        raw_doc = load_message(request.text, source_label=request.source_label)
        return _ingest_document(raw_doc, session_id=request.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Message ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.post("/auto", response_model=IngestResponse)
async def ingest_auto(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    session_id: str = Form(""),
):
    """Unified smart ingest — auto-detects input type and routes.

    Priority:
    1. If a file is uploaded → ingest as file (PDF/DOCX)
    2. If text looks like a URL → ingest as URL
    3. Otherwise → ingest as pasted text/message

    This is the single entry point for the Ingest tab's smart input.
    """
    # Priority 1: File upload
    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".pdf", ".docx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Use .pdf or .docx",
            )

        logger.info("Auto-ingest: file detected — %s", file.filename)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            raw_doc = load_pdf(tmp_path) if suffix == ".pdf" else load_docx(tmp_path)
            raw_doc.source_ref = file.filename
            raw_doc.metadata["filename"] = file.filename
            return _ingest_document(raw_doc, session_id=session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File ingest failed: {e}")
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    # Priority 2/3: Text input (URL or plain text)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide either a file or text content.",
        )

    text = text.strip()

    # Check if it looks like a URL
    if URL_REGEX.match(text):
        logger.info("Auto-ingest: URL detected — %s", text[:100])
        try:
            raw_doc = load_url(text)
            return _ingest_document(raw_doc, session_id=session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"URL ingest failed: {e}")

    # Plain text / pasted message
    logger.info("Auto-ingest: plain text detected (%d chars)", len(text))
    try:
        raw_doc = load_message(text, source_label="pasted")
        return _ingest_document(raw_doc, session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text ingest failed: {e}")


@router.get("/status")
async def ingest_status():
    """Return current ingestion/collection statistics."""
    store = _get_vector_store()
    return store.get_stats()
