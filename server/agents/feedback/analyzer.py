import logging
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from providers.nim_adapter import NIMAdapter
from .database import async_session, ImprovementTask, FineTuningCandidate
from .data_collector import DataCollector

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    def __init__(self):
        self.nim = NIMAdapter()
        self.data_collector = DataCollector()

    async def analyze_and_generate_improvements(self) -> Dict[str, Any]:
        """Analyze recent data and generate improvement tasks."""
        try:
            # Collect recent data
            transcripts = await self.data_collector.get_recent_transcripts(50)
            metrics = await self.data_collector.get_recent_metrics(50)
            feedbacks = await self.data_collector.get_feedback_data(50)

            # Prepare analysis prompt
            analysis_prompt = self._build_analysis_prompt(transcripts, metrics, feedbacks)

            # Use Mistral to analyze
            messages = [
                {"role": "system", "content": "You are an expert Aetherium system analyst. Analyze the provided data and generate specific improvement tasks for the Aetherium agent system."},
                {"role": "user", "content": analysis_prompt}
            ]

            response = self.nim.call_model(messages, temperature=0.3)
            analysis_result = response.structured_response or self._parse_analysis_response(response.text)

            # Generate improvement tasks
            improvement_tasks = await self._generate_improvement_tasks(analysis_result)

            # Generate fine-tuning candidates
            fine_tuning_candidates = await self._generate_fine_tuning_candidates(transcripts, feedbacks)

            return {
                "analysis": analysis_result,
                "improvement_tasks": improvement_tasks,
                "fine_tuning_candidates": fine_tuning_candidates,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def _build_analysis_prompt(self, transcripts: List[Dict], metrics: List[Dict], feedbacks: List[Dict]) -> str:
        """Build a comprehensive analysis prompt."""
        prompt = "Analyze the following Aetherium agent system data and identify areas for improvement:\n\n"

        prompt += "RECENT TRANSCRIPTS:\n"
        for t in transcripts[:10]:  # Limit for prompt size
            prompt += f"- Agent: {t['agent_name']}, Task: {t['task_id']}\n  Content: {t['content'][:200]}...\n"

        prompt += "\nRECENT METRICS:\n"
        for m in metrics[:20]:
            prompt += f"- {m['agent_name']}: {m['metric_name']} = {m['value']} {m.get('unit', '')}\n"

        prompt += "\nRECENT FEEDBACK:\n"
        for f in feedbacks[:10]:
            prompt += f"- Rating: {f['rating']}/10, Comments: {f.get('comments', 'N/A')}\n"

        prompt += "\nPlease provide:\n1. Key patterns and insights\n2. Specific improvement recommendations\n3. Priority levels (1-5, 5 being highest)\n4. Categories (prompt_optimization, model_fine_tuning, system_improvements, etc.)"

        return prompt

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the Mistral response into structured data."""
        # Simple parsing - in practice, might need more sophisticated parsing
        return {
            "insights": response_text,
            "recommendations": [],  # Would parse from response
            "patterns": []
        }

    async def _generate_improvement_tasks(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate and store improvement tasks based on analysis."""
        tasks = []

        # Example tasks based on analysis
        if "insights" in analysis_result:
            insights = analysis_result["insights"]
            if "performance" in insights.lower():
                task = await self._create_improvement_task(
                    "Optimize agent performance based on metric analysis",
                    4, "system_improvements", insights
                )
                tasks.append(task)

            if "prompt" in insights.lower():
                task = await self._create_improvement_task(
                    "Refine system prompts for better task understanding",
                    3, "prompt_optimization", insights
                )
                tasks.append(task)

        return tasks

    async def _create_improvement_task(self, description: str, priority: int,
                                      category: str, generated_from: str) -> Dict[str, Any]:
        """Create and store an improvement task."""
        async with async_session() as session:
            task = ImprovementTask(
                description=description,
                priority=priority,
                category=category,
                generated_from=generated_from
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)

            return {
                "id": task.id,
                "description": task.description,
                "priority": task.priority,
                "category": task.category,
                "status": task.status,
                "created_at": task.created_at.isoformat()
            }

    async def _generate_fine_tuning_candidates(self, transcripts: List[Dict],
                                             feedbacks: List[Dict]) -> List[Dict[str, Any]]:
        """Generate fine-tuning dataset candidates from successful interactions."""
        candidates = []

        # Find high-rated interactions
        high_rated_feedbacks = [f for f in feedbacks if f.get('rating', 0) >= 8]

        for feedback in high_rated_feedbacks:
            # Find corresponding transcripts
            related_transcripts = [
                t for t in transcripts
                if t['task_id'] == feedback['task_id'] and t.get('subtask_id') == feedback.get('subtask_id')
            ]

            for transcript in related_transcripts:
                # Use Mistral to extract input/output pairs
                candidate = await self._extract_fine_tuning_pair(transcript, feedback)
                if candidate:
                    candidates.append(candidate)

        return candidates

    async def _extract_fine_tuning_pair(self, transcript: Dict[str, Any],
                                       feedback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract input/output pair for fine-tuning from transcript and feedback."""
        try:
            # Simple extraction - in practice, more sophisticated parsing needed
            input_text = transcript['content']
            expected_output = feedback.get('comments', 'Successful completion')

            async with async_session() as session:
                candidate = FineTuningCandidate(
                    task_id=transcript['task_id'],
                    transcript_id=transcript['id'],
                    input_text=input_text,
                    expected_output=expected_output,
                    confidence_score=0.8,  # Based on high rating
                    tags={"source": "feedback_analysis", "rating": feedback.get('rating')}
                )
                session.add(candidate)
                await session.commit()
                await session.refresh(candidate)

                return {
                    "id": candidate.id,
                    "input_text": candidate.input_text,
                    "expected_output": candidate.expected_output,
                    "confidence_score": candidate.confidence_score,
                    "tags": candidate.tags
                }

        except Exception as e:
            logger.error(f"Error extracting fine-tuning pair: {str(e)}")
            return None