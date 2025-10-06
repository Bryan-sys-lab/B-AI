import requests
import os
from typing import List

class TestResult:
    def __init__(self, success: bool, output: str, coverage: float = 0.0):
        self.success = success
        self.output = output
        self.coverage = coverage

class Tester:
    def __init__(self):
        self.tool_api_url = os.getenv("TOOL_API_GATEWAY_URL", "http://localhost:8001")

    def run_tests(self, repo_url: str, branch: str, failing_tests: List[str]) -> TestResult:
        # Assume test_command is something like "pytest" or from failing_tests
        # For MVP, use a generic test command
        test_command = "pytest"  # placeholder

        response = requests.post(f"{self.tool_api_url}/run_tests", json={
            "repo_url": repo_url,
            "test_command": test_command,
            "branch": branch
        })

        if response.status_code == 200:
            data = response.json()
            return TestResult(success=data.get("success", False), output=data.get("output", ""), coverage=data.get("coverage", 0.0))
        else:
            return TestResult(success=False, output="Failed to run tests", coverage=0.0)