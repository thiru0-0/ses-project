"""
ADO-specific evaluation runner for Phase 2.

Evaluates the HyDE + test-case generation pipeline against the
ADO golden Q&A set (data/evaluation/ado_golden_set.json)
using the 5-dimension test case metrics.

Usage:
    python scripts/run_ado_evaluation.py
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ragpoc.evaluation.ragas_runner import run_evaluation

logger = logging.getLogger(__name__)


def main():
    """Run the ADO evaluation via ragas_runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Starting ADO Phase 2 Evaluation...")
    results = run_evaluation(eval_mode="ado")
    
    print("\n=== ADO Phase 2 Evaluation Results (5-Dimension Scoring) ===")
    
    metrics = results.get("metrics", {})
    
    metric_display = [
        ("Test Coverage", metrics.get("test_coverage", 0)),
        ("Traceability", metrics.get("answer_relevancy", 0)),
        ("Faithfulness", metrics.get("faithfulness", 0)),
        ("Structural Completeness", metrics.get("context_precision", 0)),
        ("Guardrail Behavior", metrics.get("context_recall", 0)),
    ]
    
    for name, score in metric_display:
        print(f"{name}: {score:.1%}")
        
    print(f"\nTotal pairs evaluated: {results.get('total_questions', 0)}")
    print(f"Results saved to: data/eval_results/")


if __name__ == "__main__":
    main()
