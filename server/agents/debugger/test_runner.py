import requests
import os
from typing import List, Dict, Any

class TestResult:
    def __init__(self, success: bool, output: str, errors: List[str]):
        self.success = success
        self.output = output
        self.errors = errors

class TestRunner:
    def __init__(self):
        self.tool_api_url = os.getenv("TOOL_API_GATEWAY_URL", "http://localhost:8001")

    def run_tests(self, repo_url: str, branch: str, test_command: str = "pytest") -> TestResult:
        response = requests.post(f"{self.tool_api_url}/run_tests", json={
            "repo_url": repo_url,
            "test_command": test_command,
            "branch": branch
        })

        if response.status_code == 200:
            data = response.json()
            success = data.get("success", False)
            output = data.get("output", "")
            # Parse errors from output (simple parsing)
            errors = self._parse_errors(output)
            return TestResult(success=success, output=output, errors=errors)
        else:
            return TestResult(success=False, output="Failed to run tests", errors=["Failed to run tests"])

    def _parse_errors(self, output: str) -> List[str]:
        # Simple error parsing for pytest output
        errors = []
        lines = output.split('\n')
        for line in lines:
            if 'FAILED' in line or 'ERROR' in line:
                errors.append(line.strip())
        return errors