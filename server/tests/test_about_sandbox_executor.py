import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi.testclient import TestClient
from sandbox_executor.executor import app


def test_sandbox_executor_about():
    client = TestClient(app)
    r = client.get("/about?detail=short")
    assert r.status_code == 200
    data = r.json()
    assert data["level"] == "short"
    assert "system_prompt" in data
