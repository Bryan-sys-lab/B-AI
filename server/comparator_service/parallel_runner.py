import asyncio
import requests
from pydantic import BaseModel
from comparator_service.linter import run_linter
from comparator_service.security_scanner import run_security_scan
from comparator_service.performance_checker import run_performance_check
from tool_api_gateway.models import RunTestsRequest

SANDBOX_EXECUTOR_URL = "http://localhost:8002"
TOOL_API_GATEWAY_URL = "http://localhost:8001"

class EvaluationResult(BaseModel):
    candidate_id: str
    pass_rate: float
    coverage_delta: float
    lint_score: float
    security_risk: float
    performance_impact: float
    score: float

async def run_parallel_evaluations(candidates, test_command, repo_url, branch):
    """
    Run evaluations for all candidates in parallel.
    """
    tasks = []
    for candidate in candidates:
        task = asyncio.create_task(evaluate_candidate(candidate, test_command, repo_url, branch))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results

async def evaluate_candidate(candidate, test_command, repo_url, branch):
    """
    Evaluate a single candidate: run tests, linter, security, performance.
    """
    # For simplicity, assume patch is applied in sandbox for each check
    # In reality, might need to prepare workspace with patch applied

    # Run tests
    test_result = await run_tests(candidate, test_command, repo_url, branch)

    # Run linter
    lint_result = await run_linter(candidate, repo_url, branch)

    # Run security scan
    security_result = await run_security_scan(candidate, repo_url, branch)

    # Run performance check
    performance_result = await run_performance_check(candidate, repo_url, branch)

    # Parse results to metrics
    pass_rate = parse_pass_rate(test_result)
    coverage_delta = parse_coverage_delta(test_result)
    lint_score = parse_lint_score(lint_result)
    security_risk = parse_security_risk(security_result)
    performance_impact = parse_performance_impact(performance_result)

    return EvaluationResult(
        candidate_id=candidate.id,
        pass_rate=pass_rate,
        coverage_delta=coverage_delta,
        lint_score=lint_score,
        security_risk=security_risk,
        performance_impact=performance_impact,
        score=0  # Will be calculated later
    )

async def run_tests(candidate, test_command, repo_url, branch):
    # Use tool_api_gateway run_tests
    request = RunTestsRequest(repo_url=repo_url, test_command=test_command, branch=branch)
    response = requests.post(f"{TOOL_API_GATEWAY_URL}/run_tests", json=request.model_dump())
    return response.json() if response.status_code == 200 else {"success": False, "output": ""}

# Placeholder parsers - in real implementation, parse the output
def parse_pass_rate(test_result):
    return 1.0 if test_result.get("success") else 0.0

def parse_coverage_delta(test_result):
    return 0.0  # Placeholder

def parse_lint_score(lint_result):
    return 0.0  # Placeholder

def parse_security_risk(security_result):
    return 0.0  # Placeholder

def parse_performance_impact(performance_result):
    return 0.0  # Placeholder