"""
Evaluation API routes.

POST /evaluate/run     — trigger a RAGAS evaluation run (blocking)
GET  /evaluate/results — list all past result files with their metrics
GET  /evaluate/results/latest — fetch the most recent result JSON
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

EVAL_RESULTS_DIR = Path(settings.eval_results_dir)


@router.post("/run")
async def run_evaluation():
    """Trigger a full RAGAS evaluation run against the golden Q&A set.

    Runs synchronously — may take several minutes depending on golden set size
    and LLM speed.  Returns the computed metrics and per-question details.
    """
    golden_path = Path(settings.golden_qa_path)
    if not golden_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Golden Q&A set not found at '{golden_path}'. "
                "Create data/golden_qa/golden_qa_set.jsonl first."
            ),
        )

    try:
        # Import here to avoid heavy imports at startup
        from src.ragpoc.evaluation.ragas_runner import run_evaluation as _run

        logger.info("Starting RAGAS evaluation run...")
        results = _run()
        logger.info("RAGAS evaluation complete. Metrics: %s", results.get("metrics"))
        return results

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Evaluation run failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


@router.get("/results")
async def list_results():
    """Return a list of all past evaluation result files with summary metrics."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_files = sorted(EVAL_RESULTS_DIR.glob("*.json"), reverse=True)
    if not result_files:
        return {"results": [], "total": 0}

    summaries = []
    for path in result_files[:20]:  # cap at 20 most recent
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            summaries.append(
                {
                    "filename": path.name,
                    "run_timestamp": data.get("run_timestamp", ""),
                    "total_questions": data.get("total_questions", 0),
                    "metrics": data.get("metrics", {}),
                }
            )
        except Exception:
            summaries.append({"filename": path.name, "error": "Could not parse"})

    return {"results": summaries, "total": len(result_files)}


@router.get("/results/latest")
async def get_latest_result():
    """Return the full JSON of the most recent evaluation run."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_files = sorted(EVAL_RESULTS_DIR.glob("*.json"), reverse=True)
    if not result_files:
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Run an evaluation first.",
        )

    try:
        with open(result_files[0], encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read result file: {e}"
        )
