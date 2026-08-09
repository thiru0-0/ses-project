"""
Build the vector store index from synthetic data.

Reads all files from data/synthetic/, runs them through the ingestion
pipeline (load → normalize → chunk → embed → store in ChromaDB),
and reports stats at the end.

Usage:
    python scripts/build_index.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.ragpoc.ingestion.loaders import load_pdf, load_docx, load_message
from src.ragpoc.ingestion.normalizer import normalize
from src.ragpoc.ingestion.chunker import chunk_document
from src.ragpoc.retrieval.vector_store import VectorStore
from src.ragpoc.ingestion.ado_loader import load_ado_work_items_from_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_index():
    """Build the vector store index from all synthetic data."""
    synthetic_dir = Path(settings.synthetic_data_dir)

    if not synthetic_dir.exists():
        print(f"❌ Synthetic data directory not found: {synthetic_dir}")
        print("   Run `python scripts/generate_synthetic_data.py` first.")
        sys.exit(1)

    # Initialize vector store
    print("🔧 Initializing vector store...")
    store = VectorStore()

    # Clear existing data
    print("🗑️  Clearing existing index...")
    store.clear()

    total_docs = 0
    total_chunks = 0

    # --- Process documents (PDF, DOCX, TXT) ---
    docs_dir = synthetic_dir / "documents"
    if docs_dir.exists():
        print(f"\n📄 Processing documents from {docs_dir}...")
        for filepath in sorted(docs_dir.iterdir()):
            if filepath.suffix.lower() in (".pdf", ".docx", ".txt"):
                try:
                    # Load based on type
                    if filepath.suffix.lower() == ".pdf":
                        raw_doc = load_pdf(filepath)
                    elif filepath.suffix.lower() == ".docx":
                        raw_doc = load_docx(filepath)
                    else:
                        # Plain text files — load as message type
                        content = filepath.read_text(encoding="utf-8")
                        raw_doc = load_message(content, source_label=str(filepath))
                        raw_doc.source_type = "docx"  # treat as document for chunking
                        raw_doc.metadata["filename"] = filepath.name

                    # Normalize and chunk
                    normalized = normalize(raw_doc)
                    chunks = chunk_document(normalized)

                    # Add to index
                    store.add_chunks(chunks)

                    total_docs += 1
                    total_chunks += len(chunks)
                    print(f"  ✅ {filepath.name}: {len(chunks)} chunks")

                except Exception as e:
                    print(f"  ❌ {filepath.name}: {e}")

    # --- Process chat exports ---
    chat_dir = synthetic_dir / "chat_exports"
    if chat_dir.exists():
        print(f"\n💬 Processing chat exports from {chat_dir}...")
        for filepath in sorted(chat_dir.iterdir()):
            if filepath.suffix.lower() in (".txt", ".json"):
                try:
                    content = filepath.read_text(encoding="utf-8")
                    raw_doc = load_message(content, source_label=str(filepath))
                    raw_doc.metadata["filename"] = filepath.name

                    normalized = normalize(raw_doc)
                    chunks = chunk_document(normalized)

                    store.add_chunks(chunks)

                    total_docs += 1
                    total_chunks += len(chunks)
                    print(f"  ✅ {filepath.name}: {len(chunks)} chunks")

                except Exception as e:
                    print(f"  ❌ {filepath.name}: {e}")

    # --- Process ADO Work Items ---
    ado_dir = synthetic_dir / "ado_work_items"
    if ado_dir.exists():
        print(f"\n🎫 Processing ADO work items from {ado_dir}...")
        for filepath in sorted(ado_dir.iterdir()):
            if filepath.suffix.lower() == ".json":
                try:
                    # ADO loader returns a list of RawDocuments
                    raw_docs = load_ado_work_items_from_file(filepath)
                    for raw_doc in raw_docs:
                        normalized = normalize(raw_doc)
                        chunks = chunk_document(normalized)
                        store.add_chunks(chunks)
                        total_docs += 1
                        total_chunks += len(chunks)
                    print(f"  ✅ {filepath.name}: loaded {len(raw_docs)} items")
                except Exception as e:
                    print(f"  ❌ {filepath.name}: {e}")

    # --- Report stats ---
    stats = store.get_stats()
    print(f"\n{'=' * 50}")
    print(f"📊 Index Build Complete")
    print(f"{'=' * 50}")
    print(f"  Documents processed: {total_docs}")
    print(f"  Total chunks:        {total_chunks}")
    print(f"  Collection:          {stats['collection_name']}")
    print(f"  Stored chunks:       {stats['total_chunks']}")
    print(f"  Persist directory:   {stats['persist_dir']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    build_index()
