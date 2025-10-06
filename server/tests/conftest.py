"""
Shared test fixtures and configuration for the test suite.
"""
import os
import sys
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


@pytest.fixture
def mock_adapter():
    """Mock adapter for testing provider components."""
    adapter = MagicMock()
    adapter.call_model = AsyncMock()
    adapter.api_key = "test_key"
    return adapter


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing API calls."""
    client = MagicMock()
    client.get = MagicMock()
    client.post = MagicMock()
    client.put = MagicMock()
    client.delete = MagicMock()
    return client


@pytest.fixture
def mock_async_http_client():
    """Mock async HTTP client for testing async API calls."""
    client = MagicMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "id": "test_task_123",
        "description": "Fix the bug in the fibonacci function",
        "status": "pending",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_subtask_data():
    """Sample subtask data for testing."""
    return {
        "id": "test_subtask_456",
        "task_id": "test_task_123",
        "description": "Write unit tests for fibonacci function",
        "agent_name": "testing",
        "priority": 5,
        "confidence": 0.9,
        "status": "pending"
    }


@pytest.fixture
def sample_candidate_data():
    """Sample candidate data for comparator testing."""
    return {
        "id": "candidate_001",
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "metadata": {
            "language": "python",
            "framework": "none"
        }
    }


@pytest.fixture
def mock_database_session():
    """Mock database session for testing database operations."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.query = MagicMock()
    session.filter = MagicMock()
    session.first = AsyncMock()
    session.all = AsyncMock()
    return session


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client for testing storage operations."""
    client = MagicMock()
    client.bucket_exists = MagicMock(return_value=True)
    client.make_bucket = MagicMock()
    client.fput_object = MagicMock()
    client.fget_object = MagicMock()
    client.list_objects = MagicMock(return_value=[])
    return client


@pytest.fixture
def temp_workspace():
    """Temporary workspace directory for testing file operations."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing shell command execution."""
    process = MagicMock()
    process.stdout = "command output"
    process.stderr = ""
    process.returncode = 0
    process.communicate = MagicMock(return_value=("output", "error"))
    process.wait = MagicMock()

    with pytest.mock.patch('subprocess.run', return_value=process) as mock_run:
        with pytest.mock.patch('subprocess.Popen', return_value=process) as mock_popen:
            yield {
                'run': mock_run,
                'Popen': mock_popen,
                'process': process
            }


@pytest.fixture
def mock_opentelemetry():
    """Mock OpenTelemetry components for testing observability."""
    with pytest.mock.patch('opentelemetry.trace.get_tracer') as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span = MagicMock(return_value=mock_span)
        mock_get_tracer.return_value = mock_tracer

        yield {
            'tracer': mock_tracer,
            'span': mock_span,
            'get_tracer': mock_get_tracer
        }


@pytest.fixture
def mock_prometheus():
    """Mock Prometheus metrics for testing."""
    with pytest.mock.patch('prometheus_client.Counter') as mock_counter:
        with pytest.mock.patch('prometheus_client.Histogram') as mock_histogram:
            with pytest.mock.patch('prometheus_client.Gauge') as mock_gauge:
                mock_counter_instance = MagicMock()
                mock_histogram_instance = MagicMock()
                mock_gauge_instance = MagicMock()

                mock_counter.return_value = mock_counter_instance
                mock_histogram.return_value = mock_histogram_instance
                mock_gauge.return_value = mock_gauge_instance

                yield {
                    'counter': mock_counter_instance,
                    'histogram': mock_histogram_instance,
                    'gauge': mock_gauge_instance
                }


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    original_env = os.environ.copy()

    # Set up test environment variables
    test_env = {
        'MISTRAL_API_KEY': 'test_mistral_key',
        'DEEPSEEK_API_KEY': 'test_deepseek_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'DATABASE_URL': 'sqlite:///:memory:',
        'REDIS_URL': 'redis://localhost:6379',
        'MINIO_ENDPOINT': 'localhost:9000',
        'MINIO_ACCESS_KEY': 'test_access',
        'MINIO_SECRET_KEY': 'test_secret',
        'JAEGER_HOST': 'localhost',
        'JAEGER_PORT': '6831',
        'PROMETHEUS_PORT': '9090'
    }

    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration object."""
    return {
        "test_timeout": 30,
        "mock_external_services": True,
        "cleanup_temp_files": True,
        "parallel_execution": False
    }


@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI application for testing endpoints."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    client = TestClient(app)

    yield app, client


@pytest.fixture
def sample_api_request():
    """Sample API request data."""
    return {
        "repo_url": "https://github.com/test/repo",
        "branch": "main",
        "test_command": "pytest",
        "code": "print('hello world')",
        "language": "python",
        "timeout": 30
    }


@pytest.fixture
def sample_api_response():
    """Sample API response data."""
    return {
        "success": True,
        "output": "Tests passed",
        "coverage": 95.5,
        "duration": 2.3,
        "artifacts": ["test_results.xml", "coverage.html"]
    }


@pytest.fixture
def mock_service_response():
    """Mock response from external services."""
    return {
        "status_code": 200,
        "json": {
            "result": "success",
            "data": {"key": "value"}
        },
        "text": '{"result": "success"}'
    }


# Custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "external: mark test that requires external services")


# Test utilities
def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
    """Assert that actual dict contains all key-value pairs from expected."""
    for key, value in expected.items():
        assert key in actual, f"Key '{key}' not found in actual dict"
        assert actual[key] == value, f"Value for key '{key}' does not match: expected {value}, got {actual[key]}"


def assert_response_success(response):
    """Assert that an API response indicates success."""
    assert response.status_code == 200
    data = response.json()
    assert "success" in data or "result" in data


def create_mock_model_response(text: str = "Mock response", error: str = None):
    """Create a mock ModelResponse object."""
    from providers.base_adapter import ModelResponse
    return ModelResponse(text=text, error=error)


def create_mock_evaluation_result(**kwargs):
    """Create a mock EvaluationResult object."""
    from comparator_service.parallel_runner import EvaluationResult
    defaults = {
        "candidate_id": "test_candidate",
        "pass_rate": 1.0,
        "coverage_delta": 0.0,
        "lint_score": 0.0,
        "security_risk": 0.0,
        "performance_impact": 0.0,
        "score": 1.0
    }
    defaults.update(kwargs)
    return EvaluationResult(**defaults)