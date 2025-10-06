import os
import sys

# Ensure repo root is importable for tests run from workspace root
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi.testclient import TestClient

from agents.fix_implementation.main import app


def test_about_short():
    client = TestClient(app)
    r = client.get("/about?detail=short")
    assert r.status_code == 200
    data = r.json()
    assert data["level"] == "short"
    assert "response" in data
    assert "system_prompt" in data


def test_about_medium_and_detailed():
    client = TestClient(app)
    for lvl in ("medium", "detailed"):
        r = client.get(f"/about?detail={lvl}")
        assert r.status_code == 200
        data = r.json()
        assert data["level"] == lvl
        assert isinstance(data["response"], str)
        assert "system_prompt" in data
