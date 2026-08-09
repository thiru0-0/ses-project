"""
ADO work item loader — loads synthetic or exported ADO work items from JSON files.

Produces RawDocument instances that feed into the existing
normalizer → chunker → vector store pipeline, tagged with
source_type="ado_work_item" for automatic mode detection.

When a real ADO instance is available, use mcp_client.py instead.
This loader handles the offline / synthetic-data path.
"""

import json
import logging
from pathlib import Path

from src.ragpoc.ingestion.loaders import RawDocument

logger = logging.getLogger(__name__)


def load_ado_work_items_from_file(
    path: str | Path,
) -> list[RawDocument]:
    """Load ADO work items from a JSON file and convert to RawDocuments.

    Expected JSON format:
    {
      "work_items": [
        {
          "id": 4521,
          "title": "...",
          "work_item_type": "User Story",
          "state": "Active",
          "description": "...",
          "acceptance_criteria": "...",
          "area_path": "...",
          "iteration_path": "...",
          "tags": "...",
          ...
        }
      ]
    }

    Args:
        path: Path to the JSON file.

    Returns:
        List of RawDocument objects with source_type="ado_work_item".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ADO work items file not found: {path}")

    logger.info("Loading ADO work items from: %s", path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("work_items", [])
    if not items:
        logger.warning("No work items found in %s", path)
        return []

    docs = []
    for item in items:
        content = _render_work_item_text(item)
        if not content.strip():
            continue

        docs.append(
            RawDocument(
                content=content,
                source_type="ado_work_item",
                source_ref=f"ADO#{item.get('id', 0)}: {item.get('title', 'Untitled')}",
                metadata={
                    "title": item.get("title", "Untitled"),
                    "work_item_id": item.get("id", 0),
                    "work_item_type": item.get("work_item_type", "User Story"),
                    "state": item.get("state", ""),
                    "area_path": item.get("area_path", ""),
                    "iteration_path": item.get("iteration_path", ""),
                    "tags": item.get("tags", ""),
                    "ado_source": "synthetic",
                },
            )
        )

    logger.info("Loaded %d ADO work items from %s", len(docs), path)
    return docs


def _render_work_item_text(item: dict) -> str:
    """Render a work item dict into structured plain text for embedding."""
    import re

    def strip_html(html: str) -> str:
        if not html:
            return ""
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    wi_type = item.get("work_item_type", "Work Item")
    wi_id = item.get("id", 0)
    title = item.get("title", "Untitled")

    parts = [
        f"[{wi_type}] #{wi_id}: {title}",
        f"State: {item.get('state', 'Unknown')}",
    ]

    if item.get("area_path"):
        parts.append(f"Area: {item['area_path']}")
    if item.get("iteration_path"):
        parts.append(f"Iteration: {item['iteration_path']}")
    if item.get("tags"):
        parts.append(f"Tags: {item['tags']}")
    if item.get("assigned_to"):
        parts.append(f"Assigned to: {item['assigned_to']}")
    if item.get("parent_id"):
        parts.append(f"Parent: #{item['parent_id']}")

    desc = strip_html(item.get("description", ""))
    if desc:
        parts.append(f"\nDescription:\n{desc}")

    ac = strip_html(item.get("acceptance_criteria", ""))
    if ac:
        parts.append(f"\nAcceptance Criteria:\n{ac}")

    steps = strip_html(item.get("steps", ""))
    if steps:
        parts.append(f"\nTest Steps:\n{steps}")

    return "\n".join(parts)
