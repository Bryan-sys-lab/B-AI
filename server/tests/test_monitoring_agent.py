import os
import sys
import pytest

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


class TestMonitoringAgent:
    """Test monitoring agent functionality."""

    @pytest.mark.anyio
    async def test_monitor_endpoint_system_metrics(self):
        """Test system monitoring endpoint with real data."""
        from fastapi.testclient import TestClient
        from agents.monitoring.main import app

        client = TestClient(app)
        response = client.post("/monitor", json={
            "target_service": "web-server",
            "metrics_type": "system",
            "time_range": "5m"
        })

        assert response.status_code == 200
        data = response.json()
        assert "system_metrics" in data

        # Check that real system metrics are returned
        metrics = data["system_metrics"]
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_usage" in metrics
        assert "network_io" in metrics
        assert "timestamp" in metrics

        # Validate metric ranges
        assert 0 <= metrics["cpu_percent"] <= 100
        assert 0 <= metrics["memory_percent"] <= 100
        assert 0 <= metrics["disk_usage"]["percent"] <= 100

        # Check alerts and recommendations arrays exist
        assert isinstance(data["alerts"], list)
        assert isinstance(data["recommendations"], list)

    @pytest.mark.anyio
    async def test_monitor_endpoint_different_services(self):
        """Test monitoring different service types."""
        from fastapi.testclient import TestClient
        from agents.monitoring.main import app

        client = TestClient(app)

        services = ["database", "cache", "api-gateway", "worker"]

        for service in services:
            response = client.post("/monitor", json={
                "target_service": service,
                "metrics_type": "system",
                "time_range": "1h"
            })

            assert response.status_code == 200
            data = response.json()
            assert "system_metrics" in data
            assert data["system_metrics"]["timestamp"] > 0

    @pytest.mark.anyio
    async def test_monitor_endpoint_different_time_ranges(self):
        """Test monitoring with different time ranges."""
        from fastapi.testclient import TestClient
        from agents.monitoring.main import app

        client = TestClient(app)

        time_ranges = ["5m", "1h", "24h"]

        for time_range in time_ranges:
            response = client.post("/monitor", json={
                "target_service": "test-service",
                "metrics_type": "system",
                "time_range": time_range
            })

            assert response.status_code == 200
            data = response.json()
            assert "system_metrics" in data

    @pytest.mark.anyio
    async def test_execute_monitoring_task(self):
        """Test monitoring task execution with real Aetherium."""
        from fastapi.testclient import TestClient
        from agents.monitoring.main import app

        client = TestClient(app)
        response = client.post("/execute", json={
            "description": "Set up monitoring for the microservices architecture"
        })

        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "result" in data
        assert "is_monitoring_task" in data
        assert data["is_monitoring_task"] is True

    @pytest.mark.anyio
    async def test_execute_hello_task(self):
        """Test simple hello task."""
        from fastapi.testclient import TestClient
        from agents.monitoring.main import app

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
        from agents.monitoring.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.anyio
    async def test_about_endpoint(self):
        """Test about endpoint."""
        from fastapi.testclient import TestClient
        from agents.monitoring.main import app

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
        from agents.monitoring.main import app

        client = TestClient(app)
        response = client.get("/about?detail=medium")

        assert response.status_code == 200
        data = response.json()
        assert data["level"] == "medium"

    @pytest.mark.anyio
    async def test_about_endpoint_invalid_detail(self):
        """Test about endpoint with invalid detail parameter."""
        from fastapi.testclient import TestClient
        from agents.monitoring.main import app

        client = TestClient(app)
        response = client.get("/about?detail=invalid")

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "detail must be one of" in data["error"]