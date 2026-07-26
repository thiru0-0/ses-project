"""
ADO-specific evaluation runner for Phase 2.

Evaluates the HyDE + test-case generation pipeline against the
ADO golden Q&A set (data/evaluation/ado_golden_set.json).

Unlike Phase 1's JSONL golden set, this one uses the structured JSON
format with user_story → expected_test_case pairs.

Usage:
    python scripts/run_ado_evaluation.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings

logger = logging.getLogger(__name__)


def load_ado_golden_set(path: str | None = None) -> list[dict]:
    """Load the ADO golden Q&A set.

    Args:
        path: Path to the JSON file. Defaults to settings.ado_golden_qa_path.

    Returns:
        List of dicts with user_story and expected_test_case.
    """
    filepath = Path(path or settings.ado_golden_qa_path)
    if not filepath.exists():
        raise FileNotFoundError(f"ADO golden set not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    logger.info("Loaded %d ADO golden Q&A pairs", len(pairs))
    return pairs


def run_ado_evaluation(golden_path: str | None = None) -> dict:
    """Run evaluation of the HyDE pipeline against ADO golden set.

    Steps:
    1. Load ADO golden Q&A pairs.
    2. For each user story, run through the full pipeline (HyDE + test_case mode).
    3. Compare generated test cases against expected test cases.
    4. Compute similarity-based metrics and save results.

    Args:
        golden_path: Path to ADO golden set JSON.

    Returns:
        Dictionary with metrics and per-question details.
    """
    from src.ragpoc.retrieval.vector_store import VectorStore
    from src.ragpoc.generation.pipeline import RAGPipeline
    from src.ragpoc.models.registry import get_embedding_provider

    # Load golden set
    pairs = load_ado_golden_set(golden_path)
    embed_provider = get_embedding_provider()

    # Initialize pipeline
    vector_store = VectorStore()
    pipeline = RAGPipeline(vector_store)

    details = []
    similarities = []

    for i, pair in enumerate(pairs, 1):
        user_story = pair["user_story"]
        expected = pair["expected_test_case"]

        logger.info(
            "Evaluating pair %d/%d: '%s'",
            i,
            len(pairs),
            user_story[:80],
        )

        # Run through pipeline in test_case mode
        result = pipeline.query(user_story, mode="test_case")

        # Compute embedding similarity between generated and expected
        if not result.declined:
            gen_emb = embed_provider.embed_query(result.answer)
            exp_emb = embed_provider.embed_query(expected)
            import numpy as np

            gen_vec = np.array(gen_emb)
            exp_vec = np.array(exp_emb)
            cos_sim = float(
                np.dot(gen_vec, exp_vec)
                / (np.linalg.norm(gen_vec) * np.linalg.norm(exp_vec) + 1e-10)
            )
        else:
            cos_sim = 0.0

        similarities.append(cos_sim)

        details.append(
            {
                "id": pair.get("id", f"pair_{i}"),
                "user_story": user_story,
                "expected_test_case": expected,
                "generated_test_case": result.answer,
                "declined": result.declined,
                "confidence": result.confidence,
                "similarity": cos_sim,
                "retrieved_chunks": result.retrieved_chunks,
                "relevant_chunks": result.relevant_chunks,
                "category": pair.get("category", ""),
                "mode": result.mode,
            }
        )

    # Aggregate metrics
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    decline_rate = sum(1 for d in details if d["declined"]) / len(details) if details else 0.0
    pass_rate = sum(1 for s in similarities if s >= 0.5) / len(similarities) if similarities else 0.0

    metrics = {
        "avg_cosine_similarity": round(avg_similarity, 4),
        "decline_rate": round(decline_rate, 4),
        "pass_rate_at_0.5": round(pass_rate, 4),
        "total_pairs": len(pairs),
        "hyde_enabled": settings.hyde_enabled,
        "relevance_threshold": settings.relevance_threshold,
    }

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "ado_phase2",
        "metrics": metrics,
        "details": details,
    }

    # Save results
    output_dir = Path(settings.eval_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ado_eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("ADO evaluation results saved to: %s", output_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    results = run_ado_evaluation()
    print("\n=== ADO Phase 2 Evaluation Results ===")
    print(json.dumps(results["metrics"], indent=2))
    print(f"\nTotal pairs evaluated: {results['metrics']['total_pairs']}")
    print(f"Results saved to: data/eval_results/")
