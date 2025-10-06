import logging
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database import async_session, Transcript, Metric

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        pass

    async def collect_transcript(self, task_id: str, subtask_id: Optional[str],
                                agent_name: str, content: str) -> str:
        """Store a transcript from an agent interaction."""
        async with async_session() as session:
            transcript = Transcript(
                task_id=task_id,
                subtask_id=subtask_id,
                agent_name=agent_name,
                content=content
            )
            session.add(transcript)
            await session.commit()
            await session.refresh(transcript)
            logger.info(f"Stored transcript {transcript.id} for task {task_id}")
            return transcript.id

    async def collect_metric(self, task_id: str, subtask_id: Optional[str],
                            agent_name: str, metric_name: str, value: float,
                            unit: Optional[str] = None) -> str:
        """Store a performance metric."""
        async with async_session() as session:
            metric = Metric(
                task_id=task_id,
                subtask_id=subtask_id,
                agent_name=agent_name,
                metric_name=metric_name,
                value=value,
                unit=unit
            )
            session.add(metric)
            await session.commit()
            await session.refresh(metric)
            logger.info(f"Stored metric {metric_name}={value} for task {task_id}")
            return metric.id

    async def get_recent_transcripts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent transcripts for analysis."""
        async with async_session() as session:
            result = await session.execute(
                select(Transcript).order_by(Transcript.timestamp.desc()).limit(limit)
            )
            transcripts = result.scalars().all()
            return [
                {
                    "id": t.id,
                    "task_id": t.task_id,
                    "subtask_id": t.subtask_id,
                    "agent_name": t.agent_name,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in transcripts
            ]

    async def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent metrics for analysis."""
        async with async_session() as session:
            result = await session.execute(
                select(Metric).order_by(Metric.timestamp.desc()).limit(limit)
            )
            metrics = result.scalars().all()
            return [
                {
                    "id": m.id,
                    "task_id": m.task_id,
                    "subtask_id": m.subtask_id,
                    "agent_name": m.agent_name,
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ]

    async def get_feedback_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent feedback from the shared database."""
        # Import here to avoid circular imports
        from orchestrator.database import async_session as orch_session, Feedback

        async with orch_session() as session:
            result = await session.execute(
                select(Feedback).order_by(Feedback.created_at.desc()).limit(limit)
            )
            feedbacks = result.scalars().all()
            return [
                {
                    "id": f.id,
                    "task_id": f.task_id,
                    "subtask_id": f.subtask_id,
                    "user_id": f.user_id,
                    "rating": f.rating,
                    "comments": f.comments,
                    "improvement_suggestions": f.improvement_suggestions,
                    "created_at": f.created_at.isoformat()
                }
                for f in feedbacks
            ]