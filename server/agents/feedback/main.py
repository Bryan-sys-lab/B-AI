import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible
load_dotenv()

# Ensure repo root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from .data_collector import DataCollector
from .analyzer import FeedbackAnalyzer
from .database import init_db

from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES

app = FastAPI(title="Feedback Agent")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
data_collector = DataCollector()
analyzer = FeedbackAnalyzer()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting feedback agent startup event")
    logger.info("Initializing feedback database...")
    await init_db()
    logger.info("Feedback database initialized")
    logger.info("Feedback agent startup event completed")

@app.get("/health")
def health():
    return {"status": "ok"}

class TranscriptRequest(BaseModel):
    task_id: str
    subtask_id: Optional[str] = None
    agent_name: str
    content: str

class MetricRequest(BaseModel):
    task_id: str
    subtask_id: Optional[str] = None
    agent_name: str
    metric_name: str
    value: float
    unit: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    improvement_tasks: List[Dict[str, Any]]
    fine_tuning_candidates: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None

@app.post("/collect_transcript")
async def collect_transcript(request: TranscriptRequest):
    try:
        transcript_id = await data_collector.collect_transcript(
            request.task_id, request.subtask_id, request.agent_name, request.content
        )
        return {"transcript_id": transcript_id, "success": True}
    except Exception as e:
        logger.error(f"Error collecting transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect_metric")
async def collect_metric(request: MetricRequest):
    try:
        metric_id = await data_collector.collect_metric(
            request.task_id, request.subtask_id, request.agent_name,
            request.metric_name, request.value, request.unit
        )
        return {"metric_id": metric_id, "success": True}
    except Exception as e:
        logger.error(f"Error collecting metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_feedback():
    try:
        result = await analyzer.analyze_and_generate_improvements()
        return AnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return AnalysisResponse(
            analysis={},
            improvement_tasks=[],
            fine_tuning_candidates=[],
            success=False,
            error=str(e)
        )

@app.get("/improvement_tasks")
async def get_improvement_tasks(limit: int = 50):
    try:
        from .database import async_session, ImprovementTask
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(
                select(ImprovementTask).order_by(ImprovementTask.created_at.desc()).limit(limit)
            )
            tasks = result.scalars().all()
            return [
                {
                    "id": t.id,
                    "description": t.description,
                    "priority": t.priority,
                    "category": t.category,
                    "status": t.status,
                    "generated_from": t.generated_from,
                    "created_at": t.created_at.isoformat()
                }
                for t in tasks
            ]
    except Exception as e:
        logger.error(f"Error retrieving improvement tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fine_tuning_candidates")
async def get_fine_tuning_candidates(limit: int = 50):
    try:
        from .database import async_session, FineTuningCandidate
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(
                select(FineTuningCandidate).order_by(FineTuningCandidate.created_at.desc()).limit(limit)
            )
            candidates = result.scalars().all()
            return [
                {
                    "id": c.id,
                    "task_id": c.task_id,
                    "transcript_id": c.transcript_id,
                    "input_text": c.input_text,
                    "expected_output": c.expected_output,
                    "confidence_score": c.confidence_score,
                    "tags": c.tags,
                    "created_at": c.created_at.isoformat()
                }
                for c in candidates
            ]
    except Exception as e:
        logger.error(f"Error retrieving fine-tuning candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/about")
def about(detail: Optional[str] = "short"):
    """Return a canned "about" response at three levels: short, medium, detailed."""
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {
        "level": level,
        "response": resp,
        "response": resp,
    }