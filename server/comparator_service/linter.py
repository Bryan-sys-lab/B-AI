import requests
from typing import List, Dict
from pydantic import BaseModel
from common.models import ShellExecRequest

SANDBOX_EXECUTOR_URL = "http://localhost:8002"

async def run_linter(candidate, repo_url, branch):
    """
    Run Semgrep and Bandit linters on the candidate's patched code.
    """
    # Assume patch is applied in sandbox
    # Run semgrep
    semgrep_cmd = "semgrep --config auto --json /workspace"
    semgrep_request = ShellExecRequest(command="sh", args=["-c", semgrep_cmd], working_dir="/workspace")
    semgrep_response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=semgrep_request.model_dump())

    # Run bandit
    bandit_cmd = "bandit -r /workspace -f json"
    bandit_request = ShellExecRequest(command="sh", args=["-c", bandit_cmd], working_dir="/workspace")
    bandit_response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=bandit_request.model_dump())

    # Combine results
    semgrep_issues = len(semgrep_response.json().get("results", [])) if semgrep_response.status_code == 200 else 0
    bandit_issues = len(bandit_response.json().get("results", [])) if bandit_response.status_code == 200 else 0

    total_issues = semgrep_issues + bandit_issues
    # Normalize to 0-1, assuming max issues 10 or something
    lint_score = min(total_issues / 10.0, 1.0)

    return {"lint_score": lint_score, "semgrep_issues": semgrep_issues, "bandit_issues": bandit_issues}