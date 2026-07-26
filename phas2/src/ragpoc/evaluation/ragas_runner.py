"""
Ragas evaluation runner.

Runs the golden Q&A set through the live RAG pipeline and computes
Ragas metrics: faithfulness, answer relevancy, context precision,
context recall.

Outputs results to data/eval_results/ as JSON for the Streamlit dashboard.

Usage:
    python -m src.ragpoc.evaluation.ragas_runner
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


def run_evaluation(golden_qa_path: str | None = None) -> dict:
    """Run the full Ragas evaluation against the golden Q&A set.

    Steps:
    1. Load golden Q&A pairs
    2. Run each question through the live pipeline
    3. Collect responses, contexts, and ground truths
    4. Compute Ragas metrics
    5. Save results to eval_results/

    Args:
        golden_qa_path: Path to golden set JSONL. Defaults to settings.

    Returns:
        Dictionary with metrics and per-question details.
    """
    from src.ragpoc.evaluation.golden_set import load_golden_set
    from src.ragpoc.retrieval.vector_store import VectorStore
    from src.ragpoc.generation.pipeline import RAGPipeline
    from src.ragpoc.retrieval.retriever import Retriever

    # Load golden set
    pairs = load_golden_set(golden_qa_path)
    logger.info("Loaded %d golden Q&A pairs for evaluation", len(pairs))

    # Initialize pipeline
    vector_store = VectorStore()
    pipeline = RAGPipeline(vector_store)
    retriever = Retriever(vector_store)

    # Collect data for Ragas
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    details = []

    for i, pair in enumerate(pairs, 1):
        logger.info(
            "Evaluating question %d/%d: '%s'",
            i,
            len(pairs),
            pair.question[:80],
        )

        # Run through pipeline
        result = pipeline.query(pair.question)

        # Get retrieved contexts
        retrieved = retriever.retrieve(pair.question)
        contexts = [chunk.content for chunk in retrieved]

        questions.append(pair.question)
        answers.append(result.answer)
        contexts_list.append(contexts)
        ground_truths.append(pair.ground_truth)

        details.append(
            {
                "question": pair.question,
                "ground_truth": pair.ground_truth,
                "answer": result.answer,
                "declined": result.declined,
                "confidence": result.confidence,
                "retrieved_chunks": result.retrieved_chunks,
                "relevant_chunks": result.relevant_chunks,
                "category": pair.category,
            }
        )

    # Compute Ragas metrics
    metrics = _compute_ragas_metrics(
        questions, answers, contexts_list, ground_truths
    )

    # Build results
    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(pairs),
        "metrics": metrics,
        "details": details,
    }

    # Save results
    output_dir = Path(settings.eval_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation results saved to: %s", output_path)
    logger.info("Metrics: %s", metrics)

    return results


def _compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Compute Ragas evaluation metrics.

    Args:
        questions: List of user questions.
        answers: List of generated answers.
        contexts: List of retrieved context lists (one per question).
        ground_truths: List of ground truth answers.

    Returns:
        Dictionary of metric name → score.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Build Ragas-compatible dataset
        eval_dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        # Run evaluation
        result = evaluate(
            dataset=eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        return {
            "faithfulness": float(result.get("faithfulness", 0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0)),
            "context_precision": float(result.get("context_precision", 0)),
            "context_recall": float(result.get("context_recall", 0)),
        }

    except ImportError:
        logger.warning(
            "Ragas not installed or import failed. "
            "Computing basic metrics only."
        )
        return _compute_basic_metrics(answers, ground_truths)
    except Exception as e:
        logger.error("Ragas evaluation failed: %s", e)
        return _compute_basic_metrics(answers, ground_truths)


def _compute_basic_metrics(
    answers: list[str], ground_truths: list[str]
) -> dict:
    """Compute basic fallback metrics when Ragas is unavailable.

    Simple overlap-based scoring as a rough approximation.
    """
    total = len(answers)
    if total == 0:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    # Simple word overlap metric
    overlap_scores = []
    for answer, truth in zip(answers, ground_truths):
        answer_words = set(answer.lower().split())
        truth_words = set(truth.lower().split())
        if truth_words:
            overlap = len(answer_words & truth_words) / len(truth_words)
            overlap_scores.append(overlap)
        else:
            overlap_scores.append(0.0)

    avg_overlap = sum(overlap_scores) / len(overlap_scores)

    return {
        "faithfulness": avg_overlap,
        "answer_relevancy": avg_overlap,
        "context_precision": avg_overlap,
        "context_recall": avg_overlap,
        "note": "basic_metrics_fallback",
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    results = run_evaluation()
    print("\n=== Evaluation Results ===")
    print(json.dumps(results["metrics"], indent=2))
    print(f"\nTotal questions: {results['total_questions']}")
    print(f"Results saved to: data/eval_results/")
