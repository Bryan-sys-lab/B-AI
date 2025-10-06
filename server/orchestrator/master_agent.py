import httpx
import json
from orchestrator.workflow import orchestrate_task_flow
from orchestrator.database import Subtask
from orchestrator.planner import Planner
from orchestrator.router import Router
from typing import List, Dict, Any, Optional

class MasterAgent:
    def __init__(self):
        self.planner = Planner()
        self.router = Router()
        # Use centralized endpoint management instead of hardcoded URLs
        from orchestrator.main import endpoint_manager
        self.endpoint_manager = endpoint_manager
        self.comparator_url = "http://comparator_service:8000/compare"

    async def orchestrate_task(self, task_id: str, manager=None):
        # Orchestrate task through the workflow
        return await orchestrate_task_flow(task_id, manager)

    async def execute_subtask(self, subtask: Subtask) -> Dict[str, Any]:
        # Try streaming endpoint first, fallback to regular endpoint
        stream_url = self.endpoint_manager.get_endpoint(subtask.agent_name, "stream")
        regular_url = self.endpoint_manager.get_endpoint(subtask.agent_name, "execute")

        if stream_url:
            # Use streaming endpoint
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:  # Longer timeout for streaming
                    # For streaming, we need to handle Server-Sent Events
                    response = await client.get(
                        stream_url,
                        params={"description": subtask.description},
                        timeout=300.0
                    )

                    if response.status_code == 200:
                        # Parse the streaming response
                        content = response.text
                        # Extract the final result from the stream
                        lines = content.split('\n')
                        final_result = None
                        for line in lines:
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                                    if data.get('type') == 'complete':
                                        final_result = data.get('result', {})
                                        break
                                except json.JSONDecodeError:
                                    continue

                        if final_result:
                            return final_result
                        else:
                            return {"error": "Failed to parse streaming response"}
                    else:
                        # Fallback to regular endpoint
                        if regular_url:
                            async with httpx.AsyncClient() as client:
                                response = await client.post(regular_url, json={"description": subtask.description})
                                return response.json()
                        else:
                            return {"error": "Streaming failed and no regular endpoint available"}

            except Exception as e:
                # Fallback to regular endpoint on streaming failure
                if regular_url:
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(regular_url, json={"description": subtask.description})
                            return response.json()
                    except Exception:
                        return {"error": f"Both streaming and regular execution failed: {str(e)}"}
                else:
                    return {"error": f"Streaming execution failed: {str(e)}"}
        else:
            # Use regular endpoint
            if not regular_url:
                return {"error": "Agent not available"}

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(regular_url, json={"description": subtask.description})
                    return response.json()
                except Exception:
                    return {"error": "Execution failed"}

    async def aggregate_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simple aggregation, perhaps use Mistral to synthesize
        return {"aggregated": outputs}

    async def quality_gate(self, output: Dict[str, Any]) -> bool:
        # Mock validation
        return True

    async def trigger_comparator(self, outputs: List[Dict[str, Any]]):
        async with httpx.AsyncClient() as client:
            await client.post(self.comparator_url, json={"patches": outputs})