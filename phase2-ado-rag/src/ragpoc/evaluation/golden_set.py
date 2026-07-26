"""
Golden Q&A set loader and validator.

Loads and validates data/golden_qa/golden_qa_set.jsonl for use
with the Ragas evaluation runner.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"question", "ground_truth"}
OPTIONAL_FIELDS = {"source_doc", "category"}


@dataclass
class GoldenQAPair:
    """A single golden Q&A pair for evaluation."""

    question: str
    ground_truth: str
    source_doc: str = ""
    category: str = ""


def load_golden_set(path: str | Path | None = None) -> list[GoldenQAPair]:
    """Load and validate the golden Q&A set from JSONL.

    Each line in the JSONL file should be a JSON object with at least:
    - question: str
    - ground_truth: str

    Optional fields:
    - source_doc: str
    - category: str

    Args:
        path: Path to the JSONL file. Defaults to settings.golden_qa_path.

    Returns:
        List of validated GoldenQAPair objects.

    Raises:
        FileNotFoundError: If the JSONL file doesn't exist.
        ValueError: If the file has validation errors.
    """
    filepath = Path(path or settings.golden_qa_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Golden Q&A set not found: {filepath}")

    logger.info("Loading golden Q&A set from: %s", filepath)

    pairs = []
    errors = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Validate required fields
            missing = REQUIRED_FIELDS - set(data.keys())
            if missing:
                errors.append(
                    f"Line {line_num}: Missing required fields: {missing}"
                )
                continue

            # Validate non-empty
            if not data["question"].strip():
                errors.append(f"Line {line_num}: Empty question")
                continue
            if not data["ground_truth"].strip():
                errors.append(f"Line {line_num}: Empty ground_truth")
                continue

            pairs.append(
                GoldenQAPair(
                    question=data["question"].strip(),
                    ground_truth=data["ground_truth"].strip(),
                    source_doc=data.get("source_doc", "").strip(),
                    category=data.get("category", "").strip(),
                )
            )

    if errors:
        error_msg = f"Golden set has {len(errors)} validation errors:\n"
        error_msg += "\n".join(errors[:10])
        if len(errors) > 10:
            error_msg += f"\n... and {len(errors) - 10} more"
        raise ValueError(error_msg)

    # Check for duplicate questions
    questions = [p.question for p in pairs]
    duplicates = {q for q in questions if questions.count(q) > 1}
    if duplicates:
        logger.warning(
            "Found %d duplicate questions in golden set", len(duplicates)
        )

    logger.info(
        "Loaded %d golden Q&A pairs (categories: %s)",
        len(pairs),
        set(p.category for p in pairs if p.category),
    )
    return pairs


def get_coverage_stats(pairs: list[GoldenQAPair]) -> dict:
    """Report coverage statistics for the golden set.

    Args:
        pairs: List of GoldenQAPair objects.

    Returns:
        Dictionary with coverage statistics.
    """
    categories = {}
    source_docs = set()
    for p in pairs:
        cat = p.category or "uncategorized"
        categories[cat] = categories.get(cat, 0) + 1
        if p.source_doc:
            source_docs.add(p.source_doc)

    return {
        "total_pairs": len(pairs),
        "categories": categories,
        "source_docs": sorted(source_docs),
        "avg_question_length": (
            sum(len(p.question) for p in pairs) / len(pairs) if pairs else 0
        ),
        "avg_answer_length": (
            sum(len(p.ground_truth) for p in pairs) / len(pairs) if pairs else 0
        ),
    }
