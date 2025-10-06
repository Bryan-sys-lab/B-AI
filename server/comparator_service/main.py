from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json
import os
import logging
from comparator_service.scorer import calculate_score
from comparator_service.parallel_runner import run_parallel_evaluations
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Comparator Service")

@app.get("/health")
def health():
    return {"status": "ok"}

# Config values are in comparator_service.config

class Candidate(BaseModel):
    id: str
    patch: str
    provider: str
    repo_url: str
    branch: str = "main"

class CompareCandidatesRequest(BaseModel):
    candidates: List[Candidate]
    test_command: str
    repo_url: str
    branch: str = "main"

class CompareCandidatesResponse(BaseModel):
    ranked_candidates: List[Dict]
    evidence_bundle: Dict

@app.post("/compare_candidates", response_model=CompareCandidatesResponse)
async def compare_candidates(request: CompareCandidatesRequest):
    try:
        # Run parallel evaluations
        results = await run_parallel_evaluations(request.candidates, request.test_command, request.repo_url, request.branch)

        # Calculate scores
        scored_results = []
        for result in results:
            score = calculate_score(result)
            scored_results.append({
                "candidate_id": result.candidate_id,
                "score": score,
                "details": result.model_dump()
            })

        # Rank by score descending
        ranked = sorted(scored_results, key=lambda x: x["score"], reverse=True)

        # Prepare evidence bundle
        evidence = {
            "evaluations": [r.model_dump() for r in results],
            "scoring": scored_results
        }

        # Write outputs
        output_dir = os.getenv("OUTPUT_DIR", "/output")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ranked_candidates.json"), "w") as f:
            json.dump(ranked, f, indent=2)
        with open(os.path.join(output_dir, "evidence_bundle.json"), "w") as f:
            json.dump(evidence, f, indent=2)

        return CompareCandidatesResponse(ranked_candidates=ranked, evidence_bundle=evidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute_task(request: dict):
    """Execute a comparator task"""
    try:
        description = request.get("description", "").strip()
        logger.info(f"Executing comparator task: {description}")

        # Simple canned response still allowed for trivial 'hello'
        if description.lower() == "hello":
            return {"result": "Hello, world!", "success": True}

        # For comparator tasks, delegate to the compare_candidates endpoint
        if "compare" in description.lower() or "rank" in description.lower():
            # This is a comparison request
            return {
                "result": "Comparator service is ready to compare and rank candidates. Please provide candidates to compare.",
                "success": True
            }

        # For general tasks, provide information about the service
        return {
            "result": "I am the Comparator Service. I can compare and rank solution candidates based on test results and performance metrics.",
            "success": True
        }

    except Exception as e:
        logger.error(f"Error executing comparator task: {str(e)}")
        return {"error": str(e)}

@app.get("/about")
def about(detail: str = "short"):
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}
    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {"level": level, "response": resp}