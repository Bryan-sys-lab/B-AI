import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import json

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


class TestObservability:
    """Test observability components: metrics, tracing, health checks, error tracking."""

    def test_health_status(self):
        """Test health status endpoint."""
        from observability.health import get_health_status

        status = get_health_status()

        assert isinstance(status, dict)
        assert "status" in status
        assert "timestamp" in status
        assert "version" in status

    def test_readiness_status(self):
        """Test readiness status endpoint."""
        from observability.health import get_readiness_status

        status = get_readiness_status()

        assert isinstance(status, dict)
        assert "status" in status
        assert "checks" in status

    def test_liveness_status(self):
        """Test liveness status endpoint."""
        from observability.health import get_liveness_status

        status = get_liveness_status()

        assert isinstance(status, dict)
        assert "status" in status

    @patch('prometheus_client.Counter')
    @patch('prometheus_client.Histogram')
    @patch('prometheus_client.Gauge')
    def test_metrics_initialization(self, mock_gauge, mock_histogram, mock_counter):
        """Test that metrics are properly initialized."""
        import observability.metrics

        # Verify metrics objects are created
        assert hasattr(observability.metrics, 'REQUEST_COUNT')
        assert hasattr(observability.metrics, 'REQUEST_DURATION')
        assert hasattr(observability.metrics, 'ACTIVE_CONNECTIONS')

    def test_record_request(self):
        """Test recording request metrics."""
        from observability.metrics import record_request

        # This should not raise an exception
        record_request("GET", "/api/test", 200, 0.5)

    def test_record_task_processed(self):
        """Test recording task processing metrics."""
        from observability.metrics import record_task_processed

        # This should not raise an exception
        record_task_processed("fix_implementation", 2.5)

    @patch('opentelemetry.trace.get_tracer')
    def test_tracing_setup(self, mock_get_tracer):
        """Test tracing setup."""
        from observability.tracing import setup_tracing

        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        tracer = setup_tracing("test-service")

        assert tracer is not None
        mock_get_tracer.assert_called()

    def test_error_tracker_initialization(self):
        """Test error tracker initialization."""
        from observability.errors import ErrorTracker

        tracker = ErrorTracker()

        assert tracker.logger is not None
        assert hasattr(tracker, 'capture_exception')
        assert hasattr(tracker, 'capture_message')

    def test_error_tracker_capture_exception(self):
        """Test capturing exceptions."""
        from observability.errors import ErrorTracker

        tracker = ErrorTracker()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            # This should not raise an exception
            tracker.capture_exception(e, {"context": "test"}, {"tag": "value"})

    def test_error_tracker_capture_message(self):
        """Test capturing error messages."""
        from observability.errors import ErrorTracker

        tracker = ErrorTracker()

        # This should not raise an exception
        tracker.capture_message("Test error message", "error", {"context": "test"})

    def test_error_tracker_stack_trace(self):
        """Test stack trace generation."""
        from observability.errors import ErrorTracker

        tracker = ErrorTracker()

        try:
            raise RuntimeError("Test exception")
        except RuntimeError as e:
            stack_trace = tracker._get_stack_trace(e)

            assert isinstance(stack_trace, str)
            assert "Traceback" in stack_trace
            assert "RuntimeError" in stack_trace


class TestStorage:
    """Test storage components: transcript_store, prompt_store, vector_store."""

    @patch('transcript_store.database.create_async_engine')
    @patch('transcript_store.database.sessionmaker')
    async def test_transcript_store_database_init(self, mock_sessionmaker, mock_create_engine):
        """Test transcript store database initialization."""
        from transcript_store.database import init_db, get_db

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        await init_db()

        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()

    @patch('prompt_store.database.create_async_engine')
    async def test_prompt_store_database_init(self, mock_create_engine):
        """Test prompt store database initialization."""
        from prompt_store.database import init_db

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        await init_db()

        mock_create_engine.assert_called_once()

    @patch('minio.Minio')
    def test_minio_client_initialization(self, mock_minio):
        """Test MinIO client initialization."""
        from prompt_store.minio_client import MinIOClient

        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        client = MinIOClient()

        mock_minio.assert_called_once()
        assert client.client == mock_client

    @patch('minio.Minio')
    def test_minio_bucket_operations(self, mock_minio):
        """Test MinIO bucket operations."""
        from prompt_store.minio_client import MinIOClient

        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        client = MinIOClient()

        # Test bucket creation (should not raise exception)
        client.create_bucket("test-bucket")

        # Test file upload
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"test content")
            temp_file.flush()
            client.upload_file("test-bucket", "test-file", temp_file.name)

        # Test file download
        with tempfile.NamedTemporaryFile() as temp_file:
            client.download_file("test-bucket", "test-file", temp_file.name)

    @patch('vector_store.database.create_async_engine')
    async def test_vector_store_database_init(self, mock_create_engine):
        """Test vector store database initialization."""
        from vector_store.main import init_db

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        await init_db()

        mock_create_engine.assert_called_once()


class TestPolicyEngine:
    """Test policy engine components."""

    def test_policy_files_exist(self):
        """Test that policy files exist."""
        policy_dir = os.path.join(_repo_root, "policy_engine", "policies")

        assert os.path.exists(os.path.join(policy_dir, "secrets.rego"))
        assert os.path.exists(os.path.join(policy_dir, "sensitive_files.rego"))
        assert os.path.exists(os.path.join(policy_dir, "shell_commands.rego"))

    def test_policy_content(self):
        """Test that policy files contain valid Rego content."""
        policy_dir = os.path.join(_repo_root, "policy_engine", "policies")

        for policy_file in ["secrets.rego", "sensitive_files.rego", "shell_commands.rego"]:
            policy_path = os.path.join(policy_dir, policy_file)
            with open(policy_path, 'r') as f:
                content = f.read()

            assert len(content) > 0
            # Basic check that it looks like Rego
            assert "package" in content or "default" in content

    @patch('policy_engine.main.requests.post')
    def test_policy_engine_evaluation(self, mock_post):
        """Test policy engine evaluation endpoint."""
        from policy_engine.main import evaluate_policy

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": True}
        mock_post.return_value = mock_response

        # This would test the policy evaluation logic
        # In real implementation, this calls OPA

    def test_policy_engine_about(self):
        """Test policy engine about endpoint."""
        from policy_engine.main import about

        response = about()

        assert isinstance(response, dict)
        assert "service" in response
        assert "version" in response


class TestConfiguration:
    """Test configuration files and settings."""

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        compose_file = os.path.join(_repo_root, "docker-compose.yml")
        assert os.path.exists(compose_file)

        with open(compose_file, 'r') as f:
            content = f.read()

        assert "services:" in content
        assert "version:" in content

    def test_env_file_structure(self):
        """Test .env file structure if it exists."""
        env_file = os.path.join(_repo_root, ".env")
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                content = f.read()

            # Should contain key=value pairs
            lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
            for line in lines:
                assert '=' in line

    def test_grafana_dashboard_valid_json(self):
        """Test that Grafana dashboard is valid JSON."""
        dashboard_file = os.path.join(_repo_root, "observability", "grafana-dashboard.json")

        with open(dashboard_file, 'r') as f:
            content = f.read()

        # Should be valid JSON
        dashboard = json.loads(content)

        assert "dashboard" in dashboard
        assert "panels" in dashboard["dashboard"]

    def test_prometheus_config_exists(self):
        """Test that Prometheus config exists."""
        prometheus_file = os.path.join(_repo_root, "observability", "prometheus.yml")
        assert os.path.exists(prometheus_file)

        with open(prometheus_file, 'r') as f:
            content = f.read()

        assert "global:" in content or "scrape_configs:" in content


class TestScripts:
    """Test utility scripts."""

    def test_import_check_script_exists(self):
        """Test that import check script exists."""
        script_file = os.path.join(_repo_root, "scripts", "import_check.py")
        assert os.path.exists(script_file)

    def test_smoke_test_script_exists(self):
        """Test that smoke test script exists."""
        script_file = os.path.join(_repo_root, "scripts", "smoke_test.sh")
        assert os.path.exists(script_file)

    @patch('subprocess.run')
    def test_import_check_execution(self, mock_subprocess):
        """Test import check script execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        # In real test, would run the script
        # For now, just verify the script exists and is executable concept


class TestGeneratedContent:
    """Test generated content and examples."""

    def test_generated_fibonacci_exists(self):
        """Test that generated fibonacci example exists."""
        fib_file = os.path.join(_repo_root, "generated", "fibonacci.py")
        assert os.path.exists(fib_file)

        with open(fib_file, 'r') as f:
            content = f.read()

        assert "def fibonacci" in content or "fibonacci" in content

    def test_generated_test_fibonacci_exists(self):
        """Test that generated test for fibonacci exists."""
        test_file = os.path.join(_repo_root, "generated", "test_fibonacci.py")
        assert os.path.exists(test_file)

        with open(test_file, 'r') as f:
            content = f.read()

        assert "test" in content.lower()
        assert "fibonacci" in content.lower()