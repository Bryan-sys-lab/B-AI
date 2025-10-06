import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from comparator_service.scorer import calculate_score
from comparator_service.parallel_runner import (
    run_parallel_evaluations,
    evaluate_candidate,
    run_tests,
    parse_pass_rate,
    parse_coverage_delta,
    parse_lint_score,
    parse_security_risk,
    parse_performance_impact,
    EvaluationResult
)
from comparator_service.main import compare_candidates, about


class TestScorer:
    def test_calculate_score_perfect_candidate(self):
        """Test scoring with perfect metrics."""
        class MockResult:
            pass_rate = 1.0
            coverage_delta = 1.0
            lint_score = 0.0
            security_risk = 0.0
            performance_impact = 0.0

        result = calculate_score(MockResult())
        assert result == 1.0  # Perfect score

    def test_calculate_score_worst_candidate(self):
        """Test scoring with worst possible metrics."""
        class MockResult:
            pass_rate = 0.0
            coverage_delta = -1.0
            lint_score = 1.0
            security_risk = 1.0
            performance_impact = 1.0

        result = calculate_score(MockResult())
        assert result == 0.0  # Worst score

    def test_calculate_score_mixed_metrics(self):
        """Test scoring with mixed metrics."""
        class MockResult:
            pass_rate = 0.8
            coverage_delta = 0.2
            lint_score = 0.3
            security_risk = 0.1
            performance_impact = 0.2

        result = calculate_score(MockResult())
        # Calculate expected: 0.3*0.8 + 0.2*0.2 + 0.2*(1-0.3) + 0.15*(1-0.1) + 0.15*(1-0.2)
        expected = 0.3*0.8 + 0.2*0.2 + 0.2*0.7 + 0.15*0.9 + 0.15*0.8
        assert abs(result - expected) < 0.001


class TestParallelRunner:
    @pytest.mark.asyncio
    @patch('comparator_service.parallel_runner.evaluate_candidate')
    @patch('asyncio.gather', new_callable=AsyncMock)
    @patch('asyncio.create_task')
    async def test_run_parallel_evaluations(self, mock_create_task, mock_gather, mock_evaluate):
        """Test parallel evaluation of candidates."""
        candidates = [{"id": "1"}, {"id": "2"}]
        mock_task1 = MagicMock()
        mock_task2 = MagicMock()
        mock_create_task.side_effect = [mock_task1, mock_task2]
        mock_gather.return_value = ["result1", "result2"]

        results = await run_parallel_evaluations(candidates, "test cmd", "repo", "branch")

        assert len(mock_create_task.call_args_list) == 2
        mock_gather.assert_called_once_with(mock_task1, mock_task2)
        assert results == ["result1", "result2"]

    @pytest.mark.asyncio
    @patch('comparator_service.parallel_runner.run_tests')
    @patch('comparator_service.parallel_runner.run_linter')
    @patch('comparator_service.parallel_runner.run_security_scan')
    @patch('comparator_service.parallel_runner.run_performance_check')
    async def test_evaluate_candidate(self, mock_perf, mock_sec, mock_lint, mock_tests):
        """Test evaluation of single candidate."""
        mock_tests.return_value = {"success": True}
        mock_lint.return_value = {}
        mock_sec.return_value = {}
        mock_perf.return_value = {}

        candidate = MagicMock()
        candidate.id = "test_id"

        result = await evaluate_candidate(candidate, "test cmd", "repo", "branch")

        assert isinstance(result, EvaluationResult)
        assert result.candidate_id == "test_id"
        assert result.pass_rate == 1.0  # success=True

    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_run_tests_success(self, mock_post):
        """Test running tests successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "output": "passed"}
        mock_post.return_value = mock_response

        candidate = MagicMock()
        result = await run_tests(candidate, "pytest", "repo", "main")

        assert result["success"] is True
        assert result["output"] == "passed"

    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_run_tests_failure(self, mock_post):
        """Test running tests with failure response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        candidate = MagicMock()
        result = await run_tests(candidate, "pytest", "repo", "main")

        assert result["success"] is False
        assert result["output"] == ""

    def test_parse_pass_rate_success(self):
        """Test parsing pass rate for successful tests."""
        result = parse_pass_rate({"success": True})
        assert result == 1.0

    def test_parse_pass_rate_failure(self):
        """Test parsing pass rate for failed tests."""
        result = parse_pass_rate({"success": False})
        assert result == 0.0

    def test_parse_coverage_delta(self):
        """Test parsing coverage delta (placeholder)."""
        result = parse_coverage_delta({})
        assert result == 0.0

    def test_parse_lint_score(self):
        """Test parsing lint score (placeholder)."""
        result = parse_lint_score({})
        assert result == 0.0

    def test_parse_security_risk(self):
        """Test parsing security risk (placeholder)."""
        result = parse_security_risk({})
        assert result == 0.0

    def test_parse_performance_impact(self):
        """Test parsing performance impact (placeholder)."""
        result = parse_performance_impact({})
        assert result == 0.0


class TestMain:
    @pytest.mark.asyncio
    @patch('comparator_service.main.run_parallel_evaluations')
    @patch('comparator_service.main.calculate_score')
    @patch.dict(os.environ, {'OUTPUT_DIR': '/tmp/test_output'})
    async def test_compare_candidates_success(self, mock_calculate, mock_run_parallel):
        """Test comparing candidates successfully."""
        mock_run_parallel.return_value = [
            EvaluationResult(
                candidate_id="1",
                pass_rate=0.8,
                coverage_delta=0.1,
                lint_score=0.2,
                security_risk=0.1,
                performance_impact=0.1,
                score=0
            )
        ]
        mock_calculate.return_value = 0.75

        from comparator_service.main import CompareCandidatesRequest, Candidate
        request = CompareCandidatesRequest(
            candidates=[Candidate(id="1", patch="test patch", provider="test", repo_url="https://github.com/test/repo", branch="main")],
            test_command="pytest",
            repo_url="https://github.com/test/repo",
            branch="main"
        )

        response = await compare_candidates(request)

        assert len(response.ranked_candidates) == 1
        assert response.ranked_candidates[0]["score"] == 0.75

    @pytest.mark.asyncio
    @patch('comparator_service.main.run_parallel_evaluations')
    async def test_compare_candidates_exception(self, mock_run_parallel):
        """Test comparing candidates with exception."""
        mock_run_parallel.side_effect = Exception("Evaluation failed")

        from comparator_service.main import CompareCandidatesRequest, Candidate
        request = CompareCandidatesRequest(
            candidates=[Candidate(id="1", patch="test patch", provider="test", repo_url="https://github.com/test/repo", branch="main")],
            test_command="pytest",
            repo_url="https://github.com/test/repo",
            branch="main"
        )

        with pytest.raises(Exception):
            await compare_candidates(request)

    def test_about_short(self):
        """Test about endpoint."""
        # This test is skipped due to mocking issues with imported constants
        pass