import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from agents.fix_implementation.patch_generator import PatchGenerator
from agents.fix_implementation.prompt_builder import PromptBuilder
from agents.fix_implementation.repo_manager import RepoManager
from agents.fix_implementation.safety import SafetyChecker
from agents.fix_implementation.tester import Tester, TestResult


class TestPatchGenerator:
    @patch('agents.fix_implementation.patch_generator.NIMAdapter')
    def test_init(self, mock_adapter):
        """Test PatchGenerator initialization."""
        mock_adapter.return_value = MagicMock()
        generator = PatchGenerator()
        assert generator.adapter is not None

    @patch('agents.fix_implementation.patch_generator.NIMAdapter')
    def test_generate_patches_success(self, mock_adapter_class):
        """Test successful patch generation."""
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '''```diff
diff --git a/test.py b/test.py
index 123..456 100644
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-print("hello")
+print("hello world")
```'''
        mock_adapter.call_model.return_value = mock_response
        mock_adapter_class.return_value = mock_adapter

        generator = PatchGenerator()
        patches = generator.generate_patches("Fix the print statement")

        assert len(patches) == 1
        assert 'print("hello world")' in patches[0]

    @patch('agents.fix_implementation.patch_generator.NIMAdapter')
    def test_generate_patches_json_error_fallback(self, mock_adapter_class):
        """Test patch generation with JSON parsing error."""
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.text = 'invalid response'
        mock_adapter.call_model.return_value = mock_response
        mock_adapter_class.return_value = mock_adapter

        generator = PatchGenerator()
        patches = generator.generate_patches("Fix something")

        # Should return empty list on error
        assert patches == []

    def test_extract_diffs(self):
        """Test diff extraction from text."""
        generator = PatchGenerator()
        text = '''```diff
diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old
+new
```

Some other text

```diff
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
```'''

        diffs = generator._extract_diffs(text)
        assert len(diffs) == 2
        assert 'file1.py' in diffs[0]
        assert 'file2.py' in diffs[1]


class TestPromptBuilder:
    def test_build_prompt_basic(self):
        """Test basic prompt building."""
        builder = PromptBuilder()
        failing_tests = ["test_function_returns_wrong_value"]
        prompt = builder.build_prompt(failing_tests)

        assert "failing tests" in prompt.lower()
        assert "test_function_returns_wrong_value" in prompt

    def test_build_prompt_with_context(self):
        """Test prompt building with additional context."""
        builder = PromptBuilder()
        failing_tests = ["test_something"]
        context = "The function should return the sum of inputs"
        prompt = builder.build_prompt(failing_tests, context)

        assert "test_something" in prompt
        assert "sum of inputs" in prompt


class TestRepoManager:
    def test_init(self):
        """Test RepoManager initialization."""
        manager = RepoManager()
        assert manager.github_token is None  # No token in test env

    @patch('agents.fix_implementation.repo_manager.requests.post')
    def test_apply_patch_success(self, mock_post):
        """Test successful patch application."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        manager = RepoManager()
        patch = '''diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old
+new'''
        result = manager.apply_patch("https://github.com/test/repo", "main", patch)

        assert result is True

    @patch('agents.fix_implementation.repo_manager.requests.post')
    def test_apply_patch_failure(self, mock_post):
        """Test patch application failure."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        manager = RepoManager()
        result = manager.apply_patch("https://github.com/test/repo", "main", "test patch")

        assert result is False

    def test_parse_diff_simple(self):
        """Test simple diff parsing."""
        manager = RepoManager()
        diff = '''diff --git a/test.py b/test.py
index 123..456 100644
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old line
+new line'''

        filename, content = manager._parse_diff(diff)
        assert filename == "test.py"
        assert "new line" in content

    def test_parse_diff_no_filename(self):
        """Test diff parsing with no filename."""
        manager = RepoManager()
        diff = "invalid diff format"

        filename, content = manager._parse_diff(diff)
        assert filename is None
        assert content is None


class TestSafetyChecker:
    def test_init(self):
        """Test SafetyChecker initialization."""
        checker = SafetyChecker()
        assert checker is not None

    def test_check_patch_safe(self):
        """Test checking safe patch."""
        checker = SafetyChecker()
        safe_patch = '''diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-print("hello")
+print("hello world")'''

        result = checker.check_patch(safe_patch)
        assert result is True

    def test_check_patch_with_secrets(self):
        """Test checking patch with potential secrets."""
        checker = SafetyChecker()
        unsafe_patch = '''diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1 +1 @@
-API_KEY = "old_key"
+API_KEY = "sk-1234567890abcdef"'''

        result = checker.check_patch(unsafe_patch)
        assert result is False

    def test_contains_secrets_api_key(self):
        """Test secrets detection for API keys."""
        checker = SafetyChecker()
        text_with_secret = 'API_KEY = "sk-1234567890abcdef"'
        assert checker._contains_secrets(text_with_secret) is True

    def test_contains_secrets_safe(self):
        """Test secrets detection for safe content."""
        checker = SafetyChecker()
        safe_text = 'print("hello world")'
        assert checker._contains_secrets(safe_text) is False

    @patch('agents.fix_implementation.safety.requests.post')
    def test_opa_check_success(self, mock_post):
        """Test OPA policy check success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": True}
        mock_post.return_value = mock_response

        checker = SafetyChecker()
        result = checker._opa_check("test patch")
        assert result is True

    @patch('agents.fix_implementation.safety.requests.post')
    def test_opa_check_failure(self, mock_post):
        """Test OPA policy check failure - fails open."""
        mock_post.side_effect = Exception("Connection error")

        checker = SafetyChecker()
        result = checker._opa_check("test patch")
        assert result is True  # Fails open when OPA is unavailable


class TestTester:
    def test_init(self):
        """Test Tester initialization."""
        tester = Tester()
        assert tester is not None

    @patch('agents.fix_implementation.tester.requests.post')
    def test_run_tests_success(self, mock_post):
        """Test successful test execution."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "output": "Tests passed",
            "coverage": 95.5
        }
        mock_post.return_value = mock_response

        tester = Tester()
        result = tester.run_tests("https://github.com/test/repo", "main", ["test_example"])

        assert isinstance(result, TestResult)
        assert result.success is True
        assert result.output == "Tests passed"
        assert result.coverage == 95.5

    @patch('agents.fix_implementation.tester.requests.post')
    def test_run_tests_failure(self, mock_post):
        """Test failed test execution."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "output": "Tests failed",
            "coverage": 0.0
        }
        mock_post.return_value = mock_response

        tester = Tester()
        result = tester.run_tests("https://github.com/test/repo", "main", ["test_example"])

        assert isinstance(result, TestResult)
        assert result.success is False
        assert result.output == "Tests failed"


class TestMainEndpoints:
    """Test main FastAPI endpoints for fix_implementation agent."""

    @pytest.mark.anyio
    @patch('agents.fix_implementation.main.PatchGenerator')
    @patch('agents.fix_implementation.main.PromptBuilder')
    @patch('agents.fix_implementation.main.RepoManager')
    @patch('agents.fix_implementation.main.SafetyChecker')
    @patch('agents.fix_implementation.main.Tester')
    async def test_fix_implementation_success(self, mock_tester, mock_safety, mock_repo, mock_prompt_builder, mock_patch_gen):
        """Test successful fix implementation."""
        from fastapi.testclient import TestClient
        from agents.fix_implementation.main import app

        # Mock all components
        mock_patch_gen.return_value.generate_patches.return_value = ["test patch"]
        mock_prompt_builder.return_value.build_prompt.return_value = "test prompt"
        mock_repo.return_value.apply_patch.return_value = True
        mock_safety.return_value.check_patch.return_value = True
        mock_tester.return_value.run_tests.return_value = TestResult(success=True, output="passed", coverage=100.0)

        client = TestClient(app)
        response = client.post("/fix", json={
            "repo_url": "https://github.com/test/repo",
            "failing_tests": ["test_example"],
            "context": "Fix the bug"
        })

        assert response.status_code == 200
        data = response.json()
        assert "candidate_patches" in data
        assert len(data["candidate_patches"]) > 0

    @pytest.mark.anyio
    async def test_about_endpoint(self):
        """Test about endpoint."""
        from fastapi.testclient import TestClient
        from agents.fix_implementation.main import app

        client = TestClient(app)
        response = client.get("/about")

        assert response.status_code == 200
        data = response.json()
        assert "level" in data
        assert "response" in data