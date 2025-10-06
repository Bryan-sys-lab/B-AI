import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from orchestrator.planner import Planner
from orchestrator.router import Router
from orchestrator.master_agent import MasterAgent
from orchestrator.database import Subtask

# Mock prefect modules
import sys
from unittest.mock import MagicMock
sys.modules['prefect'] = MagicMock()
sys.modules['prefect.flow'] = MagicMock()
sys.modules['prefect.task'] = MagicMock()

from orchestrator.workflow import create_subtasks_in_db, execute_subtask, aggregate_outputs, quality_gate


@pytest.fixture
def mock_adapter():
    """Mock adapter for testing."""
    adapter = MagicMock()
    adapter.call_model = AsyncMock()
    return adapter


class TestPlanner:
    @patch('orchestrator.planner.NIMAdapter')
    def test_init(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        planner = Planner()
        assert planner.adapter == mock_adapter

    @pytest.mark.anyio
    @patch('orchestrator.planner.NIMAdapter')
    async def test_decompose_task_success(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        mock_response = MagicMock()
        mock_response.text = '{"subtasks": [{"description": "test", "agent": "fix_implementation", "priority": 5, "confidence": 0.8}]}'
        mock_adapter.call_model.return_value = mock_response

        planner = Planner()
        result = await planner.decompose_task("Test task")

        assert "subtasks" in result
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["description"] == "test"

    @pytest.mark.anyio
    @patch('orchestrator.planner.NIMAdapter')
    async def test_decompose_task_json_error_fallback(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        mock_response = MagicMock()
        mock_response.text = 'invalid json'
        mock_adapter.call_model.return_value = mock_response

        planner = Planner()
        result = await planner.decompose_task("Test task")

        assert "subtasks" in result
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["agent"] == "fix_implementation"


class TestRouter:
    @patch('orchestrator.router.NIMAdapter')
    def test_init(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        router = Router()
        assert router.adapter == mock_adapter

    @pytest.mark.anyio
    @patch('orchestrator.router.NIMAdapter')
    async def test_route_subtasks_success(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        mock_response = MagicMock()
        mock_response.text = '{"agent": "fix_implementation", "confidence": 0.8, "reasoning": "test"}'
        mock_adapter.call_model.return_value = mock_response

        router = Router()
        subtasks = [{"description": "test", "agent": "fix_implementation", "priority": 5, "confidence": 0.8}]
        result = await router.route_subtasks(subtasks)

        assert len(result) == 1
        # "test" keyword routes to testing agent
        assert result[0]["agent"] == "testing"

    @pytest.mark.anyio
    @patch('orchestrator.router.NIMAdapter')
    async def test_route_subtasks_json_error_fallback(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        mock_response = MagicMock()
        mock_response.text = 'invalid json'
        mock_adapter.call_model.return_value = mock_response

        router = Router()
        subtasks = [{"description": "test", "agent": "fix_implementation", "priority": 5, "confidence": 0.8}]
        result = await router.route_subtasks(subtasks)

        # Should route to testing agent due to keyword matching fallback
        assert len(result) == 1
        assert result[0]["agent"] == "testing"


class TestMasterAgent:
    def test_init(self):
        agent = MasterAgent()
        assert agent.agent_urls == {
            "fix_implementation": "http://agent_fix_implementation:8000/execute",
        }

    @pytest.mark.anyio
    @patch('httpx.AsyncClient')
    async def test_execute_subtask_success(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.json.return_value = {"result": "success"}
        mock_client.post.return_value = mock_response

        agent = MasterAgent()
        subtask = Subtask(id="1", task_id="task1", description="test", agent_name="fix_implementation", priority=5, confidence=0.8, status="pending")

        result = await agent.execute_subtask(subtask)

        assert result == {"result": "success"}
        mock_client.post.assert_called_once_with("http://fix_implementation:8000/execute", json={"subtask": subtask.description})

    @pytest.mark.anyio
    @patch('httpx.AsyncClient')
    async def test_execute_subtask_http_error(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = Exception("Connection error")

        agent = MasterAgent()
        subtask = Subtask(id="1", task_id="task1", description="test", agent_name="fix_implementation", priority=5, confidence=0.8, status="pending")

        result = await agent.execute_subtask(subtask)

        assert "error" in result
        assert "Connection error" in result["error"]


class TestWorkflow:
    @pytest.mark.anyio
    @patch('orchestrator.workflow.async_session')
    async def test_create_subtasks_in_db(self, mock_async_session):
        mock_session = AsyncMock()
        mock_async_session.return_value.__aenter__.return_value = mock_session

        routed_subtasks = [
            {"description": "test1", "agent": "fix_implementation", "priority": 5, "confidence": 0.8},
            {"description": "test2", "agent": "debugger", "priority": 3, "confidence": 0.9}
        ]

        await create_subtasks_in_db("task1", routed_subtasks)

        # Verify subtasks were added to session
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()

    @pytest.mark.anyio
    @patch('orchestrator.workflow.httpx.AsyncClient')
    async def test_execute_subtask_workflow(self, mock_client_class):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_response = AsyncMock()
        mock_response.json.return_value = {"output": "result"}
        mock_client.post.return_value = mock_response

        subtask = Subtask(id="1", task_id="task1", description="test", agent_name="fix_implementation", priority=5, confidence=0.8, status="pending")

        result = await execute_subtask(subtask)

        assert result["output"] == "result"

    @pytest.mark.anyio
    @patch('orchestrator.workflow.NIMAdapter')
    async def test_aggregate_outputs(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        mock_response = MagicMock()
        mock_response.text = '{"aggregated": "combined output"}'
        mock_adapter.call_model.return_value = mock_response

        outputs = [
            {"result": "output1", "confidence": 0.8},
            {"result": "output2", "confidence": 0.9}
        ]

        result = await aggregate_outputs(outputs)

        assert result == {"aggregated": outputs}

    @pytest.mark.anyio
    @patch('orchestrator.workflow.NIMAdapter')
    async def test_quality_gate_pass(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        mock_response = MagicMock()
        mock_response.text = '{"approved": true}'
        mock_adapter.call_model.return_value = mock_response

        output = {"result": "good output"}

        result = await quality_gate(output)

        assert result is True

    @pytest.mark.anyio
    @patch('orchestrator.workflow.NIMAdapter')
    async def test_quality_gate_fail(self, mock_nim_adapter, mock_adapter):
        mock_nim_adapter.return_value = mock_adapter
        mock_response = MagicMock()
        mock_response.text = '{"approved": false}'
        mock_adapter.call_model.return_value = mock_response

        output = {"result": "bad output"}

        result = await quality_gate(output)

        assert result is False