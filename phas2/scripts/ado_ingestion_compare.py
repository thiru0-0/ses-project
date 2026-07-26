"""
Side-by-side comparison of ADO ingestion approaches.

Compares different scraping and chunking strategies for ADO-sourced
content to determine optimal parameters. Mirrors the Phase 1 week-2
comparison but specifically for ADO work items and wikis.

Usage:
    python scripts/ado_ingestion_compare.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings

logger = logging.getLogger(__name__)


# Sample ADO work item content for testing chunking
SAMPLE_WORK_ITEMS = [
    {
        "id": 1001,
        "type": "User Story",
        "title": "User Login with Email and Password",
        "description": (
            "As a customer, I want to be able to log in using my email "
            "and password so that I can access my account dashboard. "
            "The login form should validate the email format before "
            "submitting. The session should expire after 30 minutes of "
            "inactivity. Failed login attempts should be limited to 5 "
            "before the account is temporarily locked for 15 minutes."
        ),
        "acceptance_criteria": (
            "1. User can enter email and password on the login page.\n"
            "2. Email format is validated client-side before submission.\n"
            "3. On valid credentials, user is redirected to the dashboard.\n"
            "4. On invalid credentials, an error message is displayed.\n"
            "5. After 5 failed attempts, account is locked for 15 min.\n"
            "6. Session expires after 30 min of inactivity."
        ),
        "state": "Active",
        "area_path": "MyProject\\Authentication",
        "iteration_path": "MyProject\\Sprint 3",
    },
    {
        "id": 2001,
        "type": "Test Case",
        "title": "TC: Verify Login with Valid Credentials",
        "steps": (
            "Step 1: Navigate to https://app.example.com/login\n"
            "Step 2: Enter 'testuser@example.com' in the email field\n"
            "Step 3: Enter 'P@ssw0rd123' in the password field\n"
            "Step 4: Click the 'Sign In' button\n"
            "Expected: User is redirected to /dashboard\n"
            "Expected: Welcome message displays 'Hello, Test User'\n"
            "Step 5: Wait 30 minutes without interaction\n"
            "Expected: Session expires, user is redirected to login"
        ),
        "state": "Ready",
        "area_path": "MyProject\\Authentication",
        "iteration_path": "MyProject\\Sprint 3",
    },
]


def render_work_item_text(item: dict) -> str:
    """Render a sample work item into embeddable text."""
    parts = [
        f"[{item['type']}] #{item['id']}: {item['title']}",
        f"State: {item['state']}",
        f"Area: {item['area_path']}",
        f"Iteration: {item['iteration_path']}",
    ]
    if item.get("description"):
        parts.append(f"\nDescription:\n{item['description']}")
    if item.get("acceptance_criteria"):
        parts.append(f"\nAcceptance Criteria:\n{item['acceptance_criteria']}")
    if item.get("steps"):
        parts.append(f"\nTest Steps:\n{item['steps']}")
    return "\n".join(parts)


def compare_chunking_strategies():
    """Compare different chunk size / overlap combos on ADO content."""
    from src.ragpoc.ingestion.chunker import _recursive_split, CHARS_PER_TOKEN

    strategies = [
        {"name": "small_tight", "size_tokens": 200, "overlap_tokens": 30},
        {"name": "medium_balanced", "size_tokens": 400, "overlap_tokens": 70},
        {"name": "large_loose", "size_tokens": 600, "overlap_tokens": 100},
        {"name": "ado_default", "size_tokens": settings.chunk_size_ado, "overlap_tokens": settings.chunk_overlap_ado},
    ]

    results = []

    for item in SAMPLE_WORK_ITEMS:
        text = render_work_item_text(item)
        logger.info(
            "Processing item #%d (%s) — %d chars",
            item["id"],
            item["type"],
            len(text),
        )

        for strategy in strategies:
            chunk_size = strategy["size_tokens"] * CHARS_PER_TOKEN
            overlap = strategy["overlap_tokens"] * CHARS_PER_TOKEN

            chunks = _recursive_split(text, chunk_size, overlap)

            avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

            result = {
                "item_id": item["id"],
                "item_type": item["type"],
                "item_title": item["title"],
                "text_length": len(text),
                "strategy": strategy["name"],
                "chunk_size_tokens": strategy["size_tokens"],
                "overlap_tokens": strategy["overlap_tokens"],
                "num_chunks": len(chunks),
                "avg_chunk_length_chars": round(avg_len, 1),
                "chunks_preview": [c[:80] + "..." if len(c) > 80 else c for c in chunks],
            }
            results.append(result)
            logger.info(
                "  Strategy '%s': %d chunks, avg %.0f chars",
                strategy["name"],
                len(chunks),
                avg_len,
            )

    return results


def compare_embedding_quality():
    """Compare embedding similarity for different text representations."""
    from src.ragpoc.models.registry import get_embedding_provider
    import numpy as np

    embed = get_embedding_provider()

    # The user story and the test case — phrased very differently
    user_story = SAMPLE_WORK_ITEMS[0]
    test_case = SAMPLE_WORK_ITEMS[1]

    representations = {
        "raw_title_only": {
            "story": user_story["title"],
            "test": test_case["title"],
        },
        "full_rendered": {
            "story": render_work_item_text(user_story),
            "test": render_work_item_text(test_case),
        },
        "description_only": {
            "story": user_story.get("description", ""),
            "test": test_case.get("steps", ""),
        },
    }

    results = []
    for name, texts in representations.items():
        story_emb = np.array(embed.embed_query(texts["story"]))
        test_emb = np.array(embed.embed_query(texts["test"]))
        cos_sim = float(
            np.dot(story_emb, test_emb)
            / (np.linalg.norm(story_emb) * np.linalg.norm(test_emb) + 1e-10)
        )
        results.append(
            {
                "representation": name,
                "cosine_similarity": round(cos_sim, 4),
                "story_length": len(texts["story"]),
                "test_length": len(texts["test"]),
            }
        )
        logger.info(
            "  Representation '%s': cosine_sim=%.4f",
            name,
            cos_sim,
        )

    return results


def main():
    """Run all comparisons and save results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("ADO Ingestion Strategy Comparison")
    print("=" * 60)

    print("\n--- Chunking Strategy Comparison ---")
    chunk_results = compare_chunking_strategies()

    print("\n--- Embedding Quality Comparison ---")
    try:
        embed_results = compare_embedding_quality()
    except Exception as e:
        logger.warning("Embedding comparison skipped: %s", e)
        embed_results = []

    # Save results
    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "chunking_comparison": chunk_results,
        "embedding_comparison": embed_results,
    }

    output_dir = Path(settings.eval_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ado_ingestion_compare_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n--- Chunking Summary ---")
    print(f"{'Item':>6} | {'Strategy':<18} | {'Chunks':>6} | {'Avg Len':>8}")
    print("-" * 50)
    for r in chunk_results:
        print(
            f"#{r['item_id']:>5} | {r['strategy']:<18} | "
            f"{r['num_chunks']:>6} | {r['avg_chunk_length_chars']:>8.0f}"
        )

    if embed_results:
        print("\n--- Embedding Similarity Summary ---")
        print(f"{'Representation':<20} | {'Cosine Sim':>10}")
        print("-" * 35)
        for r in embed_results:
            print(f"{r['representation']:<20} | {r['cosine_similarity']:>10.4f}")


if __name__ == "__main__":
    main()
