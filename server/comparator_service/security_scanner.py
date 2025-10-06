import requests
from comparator_service.config import SANDBOX_EXECUTOR_URL
from tool_api_gateway.models import ShellExecRequest

async def run_security_scan(candidate, repo_url, branch):
    """
    Run security scans using Semgrep security rules and Bandit.
    """
    # Run semgrep with security config
    semgrep_cmd = "semgrep --config p/security --json /workspace"
    semgrep_request = ShellExecRequest(command="sh", args=["-c", semgrep_cmd], working_dir="/workspace")
    semgrep_response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=semgrep_request.model_dump())

    # Bandit already covers some security
    bandit_cmd = "bandit -r /workspace -f json"
    bandit_request = ShellExecRequest(command="sh", args=["-c", bandit_cmd], working_dir="/workspace")
    bandit_response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=bandit_request.model_dump())

    # Count high severity issues
    semgrep_vulns = [r for r in semgrep_response.json().get("results", []) if r.get("severity") == "high"] if semgrep_response.status_code == 200 else []
    bandit_vulns = [r for r in bandit_response.json().get("results", []) if r.get("issue_severity") == "high"] if bandit_response.status_code == 200 else []

    total_high_risks = len(semgrep_vulns) + len(bandit_vulns)
    # Normalize risk score 0-1
    security_risk = min(total_high_risks / 5.0, 1.0)  # Assume 5 high risks max

    return {"security_risk": security_risk, "high_vulnerabilities": total_high_risks}