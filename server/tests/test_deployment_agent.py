import os
import sys
import pytest
from unittest.mock import AsyncMock, patch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from providers.base_adapter import ModelResponse


class TestDeploymentAgent:
    """Test deployment agent functionality."""

    @pytest.mark.anyio
    async def test_deploy_endpoint_basic_functionality(self):
        """Test deployment endpoint basic functionality."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        client = TestClient(app)
        response = client.post("/deploy", json={
            "repo_url": "https://github.com/test/repo",
            "target_environment": "staging",
            "branch": "main",
            "deployment_type": "docker"
        })

        # Should return 200 even if Aetherium fails (fallback plan)
        assert response.status_code == 200
        data = response.json()
        assert "deployment_plan" in data
        assert data["status"] == "planned"
        assert len(data["deployment_plan"]["steps"]) > 0
        assert "estimated_duration" in data["deployment_plan"]
        assert "risk_level" in data["deployment_plan"]
        assert "rollback_plan" in data["deployment_plan"]

    @pytest.mark.anyio
    async def test_deploy_endpoint_different_environments(self):
        """Test deployment planning for different environments."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        client = TestClient(app)

        # Test production environment
        response = client.post("/deploy", json={
            "repo_url": "https://github.com/test/repo",
            "target_environment": "production",
            "branch": "main",
            "deployment_type": "kubernetes"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "planned"

        # Test development environment
        response = client.post("/deploy", json={
            "repo_url": "https://github.com/test/repo",
            "target_environment": "development",
            "branch": "develop",
            "deployment_type": "docker"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "planned"

    @pytest.mark.anyio
    async def test_execute_deployment_task(self):
        """Test deployment task execution."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        # Mock the NIMAdapter to return a successful response
        mock_response = ModelResponse(
            text='{"steps": ["Pre-deployment checks", "Build application", "Deploy to production"], "estimated_duration": 30, "risk_level": "medium", "rollback_plan": ["Rollback deployment", "Verify system health"]}',
            tokens=150,
            tool_calls=[],
            structured_response={},
            confidence=0.9,
            latency_ms=2000
        )

        with patch('agents.deployment.main.NIMAdapter') as mock_adapter_class:
            mock_adapter_instance = mock_adapter_class.return_value
            mock_adapter_instance.call_model.return_value = mock_response

            client = TestClient(app)
            response = client.post("/execute", json={
                "description": "Deploy the web application to production with zero downtime"
            })

            assert response.status_code == 200
            data = response.json()
            assert "success" in data
            assert "result" in data
            assert "is_deployment_task" in data
            assert data["is_deployment_task"] is True

    @pytest.mark.anyio
    async def test_execute_hello_task(self):
        """Test simple hello task."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        client = TestClient(app)
        response = client.post("/execute", json={
            "description": "hello"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"] == "Hello, world!"

    @pytest.mark.anyio
    async def test_health_endpoint(self):
        """Test health endpoint."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.anyio
    async def test_about_endpoint(self):
        """Test about endpoint."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        client = TestClient(app)
        response = client.get("/about")

        assert response.status_code == 200
        data = response.json()
        assert "level" in data
        assert "response" in data
        assert "system_prompt" in data

    @pytest.mark.anyio
    async def test_about_endpoint_with_detail(self):
        """Test about endpoint with detail parameter."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        client = TestClient(app)
        response = client.get("/about?detail=detailed")

        assert response.status_code == 200
        data = response.json()
        assert data["level"] == "detailed"

    @pytest.mark.anyio
    async def test_about_endpoint_invalid_detail(self):
        """Test about endpoint with invalid detail parameter."""
        from fastapi.testclient import TestClient
        from agents.deployment.main import app

        client = TestClient(app)
        response = client.get("/about?detail=invalid")

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "detail must be one of" in data["error"]