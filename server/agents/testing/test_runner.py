import os
import tempfile
import logging
from typing import Dict, List, Optional
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TestResult(BaseModel):
    passed: int
    failed: int
    errors: int
    skipped: int
    total: int
    coverage: Optional[float] = None
    stdout: str
    stderr: str
    exit_code: int
    test_files: List[str] = []

class TestRunner:
    def __init__(self, sandbox_executor_url: str = "http://localhost:8002"):
        self.sandbox_url = sandbox_executor_url

    def run_unit_tests(self, code_path: str, test_files: List[str] = None) -> TestResult:
        """Run unit tests with coverage analysis."""
        return self._run_pytest(code_path, test_files, ["--cov", "--cov-report=term-missing"])

    def run_integration_tests(self, code_path: str, test_files: List[str] = None) -> TestResult:
        """Run integration tests."""
        return self._run_pytest(code_path, test_files, ["-m", "integration"])

    def run_fuzz_tests(self, code_path: str, test_files: List[str] = None) -> TestResult:
        """Run fuzz tests."""
        return self._run_pytest(code_path, test_files, ["-m", "fuzz"])

    def run_performance_tests(self, code_path: str, test_files: List[str] = None) -> TestResult:
        """Run performance tests."""
        return self._run_pytest(code_path, test_files, ["-m", "performance"])

    def _run_pytest(self, code_path: str, test_files: List[str], extra_args: List[str] = None) -> TestResult:
        """Run pytest in sandbox with given arguments."""
        try:
            # Prepare command
            cmd_args = ["pytest"]
            if extra_args:
                cmd_args.extend(extra_args)
            if test_files:
                cmd_args.extend(test_files)
            else:
                cmd_args.append(".")

            command = " ".join(cmd_args)

            # Execute in sandbox
            payload = {
                "command": "sh",
                "args": ["-c", command],
                "working_dir": "/workspace",
                "env": {
                    "PYTHONPATH": "/workspace",
                    "PYTHONDONTWRITEBYTECODE": "1"
                },
                "timeout": 300  # 5 minutes
            }

            response = requests.post(f"{self.sandbox_url}/execute", json=payload, timeout=310)
            response.raise_for_status()
            result = response.json()

            # Parse pytest output
            return self._parse_pytest_output(result)

        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return TestResult(
                passed=0, failed=0, errors=1, skipped=0, total=0,
                stdout="", stderr=str(e), exit_code=1
            )

    def _parse_pytest_output(self, sandbox_result: Dict) -> TestResult:
        """Parse pytest output from sandbox execution."""
        stdout = sandbox_result.get("stdout", "")
        stderr = sandbox_result.get("stderr", "")
        exit_code = sandbox_result.get("exit_code", 1)

        # Simple parsing of pytest output
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        coverage = None

        # Look for summary line like "5 passed, 2 failed, 1 error, 3 skipped"
        for line in stdout.split('\n'):
            if 'passed' in line and 'failed' in line:
                # Parse the summary
                parts = line.replace(',', '').split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            if 'passed' in parts[i+1]:
                                passed = count
                            elif 'failed' in parts[i+1]:
                                failed = count
                            elif 'error' in parts[i+1]:
                                errors = count
                            elif 'skipped' in parts[i+1]:
                                skipped = count

        total = passed + failed + errors + skipped

        # Extract coverage percentage
        for line in stdout.split('\n'):
            if 'TOTAL' in line and '%' in line:
                try:
                    coverage_str = line.split()[-1].rstrip('%')
                    coverage = float(coverage_str)
                except:
                    pass

        return TestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total=total,
            coverage=coverage,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )