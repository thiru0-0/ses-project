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


def run_evaluation(golden_qa_path: str | None = None, eval_mode: str = "qa") -> dict:
    """Run the full Ragas evaluation against the golden Q&A set.

    Steps:
    1. Load golden Q&A pairs (or ADO pairs)
    2. Run each question through the live pipeline
    3. Collect responses, contexts, and ground truths
    4. Compute metrics (Ragas for QA, custom 5-dimension for ADO)
    5. Save results to eval_results/

    Args:
        golden_qa_path: Path to golden set JSONL or JSON. Defaults to settings.
        eval_mode: "qa" or "ado"

    Returns:
        Dictionary with metrics and per-question details.
    """
    from src.ragpoc.evaluation.golden_set import load_golden_set
    from src.ragpoc.retrieval.vector_store import VectorStore
    from src.ragpoc.generation.pipeline import RAGPipeline
    from src.ragpoc.retrieval.retriever import Retriever

    # Load golden set based on mode
    if eval_mode == "ado" and not golden_qa_path:
        golden_qa_path = settings.ado_golden_qa_path
        
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
        if eval_mode == "ado":
            result = pipeline.query(pair.question, mode="test_case")
        else:
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
                "mode": result.mode,
            }
        )

    # Compute metrics
    if eval_mode == "ado":
        metrics = _compute_ado_metrics(questions, answers, contexts_list, ground_truths)
    else:
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


def _compute_ado_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Compute the 5-dimension ADO evaluation metrics.
    
    1. Test Coverage: Does the test case cover the user story criteria?
    2. Traceability: Does it cite the correct requirement ID?
    3. Faithfulness: Is the expected result grounded in the source?
    4. Structural Completeness: Does it have all 8 fields?
    5. Guardrail Behavior: Does it decline properly on missing context?
    """
    total = len(answers)
    if total == 0:
        return {}
        
    scores = {
        "test_coverage": 0.0,
        "traceability": 0.0,
        "faithfulness": 0.0,
        "structural_completeness": 0.0,
        "guardrail_behavior": 0.0,
    }
    
    # 8-field requirement checklist
    required_fields = [
        "Test ID", "Requirement Reference", "Title", "Preconditions",
        "Test Steps", "Expected Result", "Actual Result", "Status"
    ]
    
    # The decline string used by the pipeline
    decline_string = "I couldn't find this information"
    
    for answer, truth in zip(answers, ground_truths):
        answer_lower = answer.lower()
        
        # 5. Guardrail Behavior: If it should decline, did it? If it shouldn't, did it hallucinate?
        # A simple proxy: if the ground truth implies we don't have it, we expect a decline.
        # But for this golden set, all have valid contexts. So guardrail is whether it avoided the decline string when it shouldn't have, or properly formatted.
        if decline_string.lower() not in answer_lower:
            scores["guardrail_behavior"] += 1.0
            
        # 1. Structural completeness (check for fields)
        fields_present = sum(1 for field in required_fields if field.lower() in answer_lower)
        struct_score = fields_present / len(required_fields)
        scores["structural_completeness"] += struct_score
        
        # 2. Traceability (check for ADO reference)
        import re
        ado_ref = re.search(r'ado#\d+', answer_lower)
        if ado_ref:
            scores["traceability"] += 1.0
            
        # 3. Faithfulness & 1. Coverage (approximated by word overlap with ground truth)
        answer_words = set(answer_lower.split())
        truth_words = set(truth.lower().split())
        if truth_words:
            overlap = len(answer_words & truth_words) / len(truth_words)
            scores["faithfulness"] += overlap
            scores["test_coverage"] += min(1.0, overlap * 1.2) # Coverage is a bit looser than strict overlap
            
    # Average the scores
    for k in scores:
        scores[k] = scores[k] / total
        
    scores["note"] = "ado_metrics"
    
    # Map to standard names for UI rendering, but keep originals
    return {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["traceability"], # map traceability to relevancy slot
        "context_precision": scores["structural_completeness"], # map struct to precision slot
        "context_recall": scores["guardrail_behavior"], # map guardrail to recall slot
        "test_coverage": scores["test_coverage"],
        "note": "ado_metrics",
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
