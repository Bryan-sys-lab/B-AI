import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi.testclient import TestClient
from tool_api_gateway.main import app


def test_tool_api_gateway_about():
    client = TestClient(app)
    r = client.get("/about?detail=short")
    assert r.status_code == 200
    data = r.json()
    assert data["level"] == "short"
    assert "system_prompt" in data


def test_run_code_python():
    client = TestClient(app)
    r = client.post("/run_code", json={
        "code": "print('Hello, World!')",
        "language": "python",
        "timeout": 10
    })
    assert r.status_code == 200
    data = r.json()
    assert "stdout" in data
    # Note: In test environment, sandbox_executor may not be available,
    # so we check that the endpoint responds correctly with error
    if "error" in data:
        assert "sandbox_executor" in data["error"] or "connection" in data["error"].lower()
    else:
        assert "Hello, World!" in data["stdout"]
        assert data["exit_code"] == 0


def test_run_code_javascript():
    client = TestClient(app)
    r = client.post("/run_code", json={
        "code": "console.log('Hello, JS!');",
        "language": "javascript",
        "timeout": 10
    })
    assert r.status_code == 200
    data = r.json()
    assert "stdout" in data
    # Note: In test environment, sandbox_executor may not be available,
    # so we check that the endpoint responds correctly with error
    if "error" in data:
        assert "sandbox_executor" in data["error"] or "connection" in data["error"].lower()
    else:
        assert "Hello, JS!" in data["stdout"]
        assert data["exit_code"] == 0


def test_run_code_unsupported_language():
    client = TestClient(app)
    r = client.post("/run_code", json={
        "code": "print('test')",
        "language": "unsupported",
        "timeout": 10
    })
    assert r.status_code == 200
    data = r.json()
    assert "error" in data
    assert "Unsupported language" in data["error"]
