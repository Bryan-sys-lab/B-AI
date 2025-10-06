import os
import sys
import pytest
from unittest.mock import patch, MagicMock

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tool_api_gateway.models import (
    GitReadFileRequest, GitReadFileResponse,
    GitWriteFileRequest, GitWriteFileResponse,
    ListFilesRequest, ListFilesResponse,
    RunTestsRequest, RunTestsResponse,
    ShellExecRequest, ShellExecResponse,
    CreatePrRequest, CreatePrResponse,
    ScanVulnRequest, ScanVulnResponse,
    SearchDocsRequest, SearchDocsResponse,
    FetchMetricsRequest, FetchMetricsResponse,
    RunCodeRequest, RunCodeResponse
)
from tool_api_gateway.security import is_command_allowed, check_opa_policy, validate_request


class TestModels:
    def test_git_read_file_request(self):
        """Test GitReadFileRequest model."""
        req = GitReadFileRequest(repo_url="https://github.com/test/repo", file_path="README.md", branch="main")
        assert req.repo_url == "https://github.com/test/repo"
        assert req.file_path == "README.md"
        assert req.branch == "main"

    def test_git_read_file_response(self):
        """Test GitReadFileResponse model."""
        resp = GitReadFileResponse(content="file content", error=None)
        assert resp.content == "file content"
        assert resp.error is None

    def test_git_write_file_request(self):
        """Test GitWriteFileRequest model."""
        req = GitWriteFileRequest(
            repo_url="https://github.com/test/repo",
            file_path="test.txt",
            content="new content",
            branch="feature",
            commit_message="Update test file"
        )
        assert req.repo_url == "https://github.com/test/repo"
        assert req.file_path == "test.txt"
        assert req.content == "new content"
        assert req.branch == "feature"
        assert req.commit_message == "Update test file"

    def test_list_files_request(self):
        """Test ListFilesRequest model."""
        req = ListFilesRequest(repo_url="https://github.com/test/repo", path="src", branch="main")
        assert req.repo_url == "https://github.com/test/repo"
        assert req.path == "src"
        assert req.branch == "main"

    def test_run_tests_request(self):
        """Test RunTestsRequest model."""
        req = RunTestsRequest(repo_url="https://github.com/test/repo", test_command="pytest", branch="main")
        assert req.repo_url == "https://github.com/test/repo"
        assert req.test_command == "pytest"
        assert req.branch == "main"

    def test_shell_exec_request(self):
        """Test ShellExecRequest model."""
        req = ShellExecRequest(
            command="ls",
            args=["-la"],
            working_dir="/tmp",
            env={"PATH": "/usr/bin"},
            timeout=60
        )
        assert req.command == "ls"
        assert req.args == ["-la"]
        assert req.working_dir == "/tmp"
        assert req.env == {"PATH": "/usr/bin"}
        assert req.timeout == 60

    def test_create_pr_request(self):
        """Test CreatePrRequest model."""
        req = CreatePrRequest(
            repo_url="https://github.com/test/repo",
            title="Test PR",
            body="Description",
            head_branch="feature",
            base_branch="main"
        )
        assert req.repo_url == "https://github.com/test/repo"
        assert req.title == "Test PR"
        assert req.body == "Description"
        assert req.head_branch == "feature"
        assert req.base_branch == "main"

    def test_scan_vuln_request(self):
        """Test ScanVulnRequest model."""
        req = ScanVulnRequest(repo_url="https://github.com/test/repo", branch="main")
        assert req.repo_url == "https://github.com/test/repo"
        assert req.branch == "main"

    def test_search_docs_request(self):
        """Test SearchDocsRequest model."""
        req = SearchDocsRequest(query="how to test", repo_url="https://github.com/test/repo")
        assert req.query == "how to test"
        assert req.repo_url == "https://github.com/test/repo"

    def test_fetch_metrics_request(self):
        """Test FetchMetricsRequest model."""
        req = FetchMetricsRequest(repo_url="https://github.com/test/repo", metric_type="coverage")
        assert req.repo_url == "https://github.com/test/repo"
        assert req.metric_type == "coverage"

    def test_run_code_request(self):
        """Test RunCodeRequest model."""
        req = RunCodeRequest(code="print('hello')", language="python", timeout=30)
        assert req.code == "print('hello')"
        assert req.language == "python"
        assert req.timeout == 30


class TestSecurity:
    def test_is_command_allowed_allowed(self):
        """Test allowed commands."""
        assert is_command_allowed("ls") is True
        assert is_command_allowed("cat") is True
        assert is_command_allowed("echo") is True

    def test_is_command_allowed_not_allowed(self):
        """Test not allowed commands."""
        assert is_command_allowed("rm") is False  # rm is in list but let's test a truly forbidden one
        assert is_command_allowed("sudo") is False
        assert is_command_allowed("curl") is False

    def test_is_command_allowed_with_args(self):
        """Test command with arguments."""
        assert is_command_allowed("ls -la") is True
        assert is_command_allowed("cat file.txt") is True

    def test_check_opa_policy(self):
        """Test OPA policy check (mock implementation)."""
        result = check_opa_policy({"test": "data"}, "test_policy")
        assert result is True  # Mock always returns True

    def test_validate_request_shell_exec_allowed(self):
        """Test validation for allowed shell_exec command."""
        result = validate_request("shell_exec", {"command": "ls"})
        assert result is True

    def test_validate_request_shell_exec_not_allowed(self):
        """Test validation for not allowed shell_exec command."""
        result = validate_request("shell_exec", {"command": "sudo"})
        assert result is False

    def test_validate_request_other_tool(self):
        """Test validation for other tools."""
        result = validate_request("git_read", {"repo_url": "test"})
        assert result is True  # OPA mock returns True


class TestMainEndpoints:
    """Test main FastAPI endpoints beyond the existing API tests."""

    @patch('tool_api_gateway.main.requests.post')
    def test_run_tests_endpoint_success(self, mock_post):
        """Test run_tests endpoint with mocked external service."""
        from fastapi.testclient import TestClient
        from tool_api_gateway.main import app

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "Tests passed", "success": True}
        mock_post.return_value = mock_response

        client = TestClient(app)
        response = client.post("/run_tests", json={
            "repo_url": "https://github.com/test/repo",
            "test_command": "pytest",
            "branch": "main"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["output"] == "Tests passed"
        assert data["success"] is True

    @patch('tool_api_gateway.main.requests.post')
    def test_run_tests_endpoint_failure(self, mock_post):
        """Test run_tests endpoint with service failure."""
        from fastapi.testclient import TestClient
        from tool_api_gateway.main import app

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        client = TestClient(app)
        response = client.post("/run_tests", json={
            "repo_url": "https://github.com/test/repo",
            "test_command": "pytest",
            "branch": "main"
        })

        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    @patch('tool_api_gateway.main.validate_request')
    def test_shell_exec_security_check(self, mock_validate):
        """Test shell_exec endpoint with security validation."""
        from fastapi.testclient import TestClient
        from tool_api_gateway.main import app

        mock_validate.return_value = False  # Command not allowed

        client = TestClient(app)
        response = client.post("/shell_exec", json={
            "command": "sudo",
            "args": ["rm", "-rf", "/"]
        })

        assert response.status_code == 403
        data = response.json()
        assert "not allowed" in data["detail"].lower()

    @patch('tool_api_gateway.main.validate_request')
    @patch('tool_api_gateway.main.requests.post')
    def test_shell_exec_success(self, mock_post, mock_validate):
        """Test shell_exec endpoint success."""
        from fastapi.testclient import TestClient
        from tool_api_gateway.main import app

        mock_validate.return_value = True
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "output",
            "stderr": "",
            "exit_code": 0,
            "artifacts": []
        }
        mock_post.return_value = mock_response

        client = TestClient(app)
        response = client.post("/shell_exec", json={
            "command": "ls",
            "args": ["-la"]
        })

        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "output"
        assert data["exit_code"] == 0