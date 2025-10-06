import os
import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock
import tempfile

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sandbox_executor.executor import execute_in_sandbox, ShellExecRequest, ShellExecResponse, execute, about
from sandbox_executor.utils import capture_artifacts, prepare_workspace, cleanup_workspace


class TestUtils:
    def test_prepare_workspace(self):
        """Test workspace preparation creates a temporary directory."""
        workspace = prepare_workspace()
        assert os.path.exists(workspace)
        assert workspace.startswith(tempfile.gettempdir())
        assert "sandbox_workspace_" in workspace

        # Cleanup
        cleanup_workspace(workspace)

    def test_cleanup_workspace(self):
        """Test workspace cleanup removes the directory."""
        workspace = prepare_workspace()
        assert os.path.exists(workspace)

        cleanup_workspace(workspace)
        assert not os.path.exists(workspace)

    def test_cleanup_workspace_nonexistent(self):
        """Test cleanup doesn't fail for non-existent directory."""
        cleanup_workspace("/nonexistent/path")
        # Should not raise exception

    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    def test_capture_artifacts_success(self, mock_temp_dir, mock_subprocess):
        """Test successful artifact capture."""
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
        mock_temp_dir.return_value.__exit__.return_value = None

        # Mock os.walk to return some files
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [
                ('/tmp/test', [], ['file1.txt', 'file2.py'])
            ]

            artifacts = capture_artifacts("container123")

            assert len(artifacts) == 2
            assert "file1.txt" in artifacts
            assert "file2.py" in artifacts

            # Verify docker cp was called
            mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_capture_artifacts_failure(self, mock_subprocess):
        """Test artifact capture handles docker cp failure."""
        mock_subprocess.side_effect = Exception("Docker error")

        artifacts = capture_artifacts("container123")

        assert artifacts == []


class TestExecutor:
    @patch('sandbox_executor.executor.prepare_workspace')
    @patch('sandbox_executor.executor.cleanup_workspace')
    @patch('subprocess.run')
    def test_execute_in_sandbox_success(self, mock_subprocess, mock_cleanup, mock_prepare):
        """Test successful command execution in sandbox."""
        mock_prepare.return_value = "/tmp/workspace"
        mock_result = MagicMock()
        mock_result.stdout = "Hello World"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        response = execute_in_sandbox("echo", ["hello"], "/workspace", {}, 30)

        assert response.stdout == "Hello World"
        assert response.stderr == ""
        assert response.exit_code == 0
        assert response.error is None

        mock_prepare.assert_called_once()
        mock_cleanup.assert_called_once_with("/tmp/workspace")

    @patch('sandbox_executor.executor.prepare_workspace')
    @patch('sandbox_executor.executor.cleanup_workspace')
    @patch('subprocess.run')
    def test_execute_in_sandbox_timeout(self, mock_subprocess, mock_cleanup, mock_prepare):
        """Test command execution timeout."""
        mock_prepare.return_value = "/tmp/workspace"
        mock_subprocess.side_effect = subprocess.TimeoutExpired("timeout", 30)

        response = execute_in_sandbox("sleep", ["60"], "/workspace", {}, 30)

        assert response.stdout == ""
        assert "timed out" in response.stderr.lower()
        assert response.exit_code == 1
        assert response.error == "Timeout"

    @patch('sandbox_executor.executor.prepare_workspace')
    @patch('sandbox_executor.executor.cleanup_workspace')
    def test_execute_in_sandbox_exception(self, mock_cleanup, mock_prepare):
        """Test command execution with general exception."""
        mock_prepare.return_value = "/tmp/workspace"

        with patch('subprocess.run', side_effect=Exception("Docker error")):
            response = execute_in_sandbox("invalid", [], "/workspace", {}, 30)

        assert response.stdout == ""
        assert response.stderr == ""
        assert response.exit_code == 1
        assert response.error == "Docker error"

    @patch('sandbox_executor.executor.execute_in_sandbox')
    def test_execute_endpoint_success(self, mock_execute):
        """Test execute endpoint success."""
        mock_response = ShellExecResponse(
            stdout="output",
            stderr="",
            exit_code=0,
            artifacts=[]
        )
        mock_execute.return_value = mock_response

        request = ShellExecRequest(command="echo", args=["hello"], timeout=30)
        response = execute(request)

        assert response.stdout == "output"
        assert response.exit_code == 0

    @patch('sandbox_executor.executor.execute_in_sandbox')
    def test_execute_endpoint_timeout_clamp(self, mock_execute):
        """Test execute endpoint clamps timeout to MAX_TIMEOUT."""
        mock_response = ShellExecResponse(
            stdout="output",
            stderr="",
            exit_code=0,
            artifacts=[]
        )
        mock_execute.return_value = mock_response

        # Request timeout > MAX_TIMEOUT (300)
        request = ShellExecRequest(command="echo", args=["hello"], timeout=600)
        execute(request)

        # Should call execute_in_sandbox with clamped timeout
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert call_args[1]['timeout'] == 300  # MAX_TIMEOUT

    @patch('sandbox_executor.executor.CANNED_RESPONSES')
    @patch('sandbox_executor.executor.SYSTEM_PROMPT')
    def test_about_short(self, mock_system_prompt, mock_responses):
        """Test about endpoint with short detail."""
        mock_responses.get.return_value = "Short response"
        mock_system_prompt.return_value = "System prompt"

        response = about("short")

        assert response["level"] == "short"
        assert response["response"] == "Short response"
        assert response["system_prompt"] == "System prompt"

    def test_about_invalid_detail(self):
        """Test about endpoint with invalid detail level."""
        response = about("invalid")

        assert "error" in response
        assert "detail must be one of" in response["error"]

    def test_about_default(self):
        """Test about endpoint with default detail."""
        with patch('sandbox_executor.executor.CANNED_RESPONSES') as mock_responses:
            mock_responses.get.return_value = "Default response"

            response = about()

            assert response["level"] == "short"