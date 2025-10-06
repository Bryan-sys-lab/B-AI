import os
import logging
from typing import Dict, List, Optional
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class LoadTestResult(BaseModel):
    total_requests: int
    requests_per_second: float
    response_time_avg: float
    response_time_95p: float
    response_time_99p: float
    failures: int
    stdout: str
    stderr: str
    exit_code: int

class LoadTester:
    def __init__(self, sandbox_executor_url: str = "http://localhost:8002"):
        self.sandbox_url = sandbox_executor_url

    def run_load_tests(self, test_code: str, target_url: str = "http://localhost:8000", users: int = 10, duration: int = 60) -> LoadTestResult:
        """Run Locust load tests in sandbox."""
        try:
            # Create locustfile.py content
            locustfile_content = f"""
from locust import HttpUser, task, between

class TestUser(HttpUser):
    wait_time = between(1, 3)

    {test_code}

# Run with: locust -f locustfile.py --host={target_url} --users={users} --spawn-rate=1 --run-time={duration}s --csv=results
"""

            # Create test script to run locust
            test_script = f"""
import subprocess
import sys
import os

# Write locustfile
with open('locustfile.py', 'w') as f:
    f.write('''{locustfile_content}''')

# Run locust in headless mode
cmd = [
    'locust', '-f', 'locustfile.py',
    '--host={target_url}',
    '--users={users}',
    '--spawn-rate=1',
    '--run-time={duration}s',
    '--csv=results',
    '--headless'
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout={duration + 30})
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
sys.exit(result.returncode)
"""

            # Execute in sandbox
            payload = {
                "command": "python3",
                "args": ["-c", test_script],
                "working_dir": "/workspace",
                "env": {
                    "PYTHONPATH": "/workspace"
                },
                "timeout": duration + 60  # Extra time for setup
            }

            response = requests.post(f"{self.sandbox_url}/execute", json=payload, timeout=duration + 70)
            response.raise_for_status()
            result = response.json()

            # Parse results
            return self._parse_load_output(result)

        except Exception as e:
            logger.error(f"Error running load tests: {str(e)}")
            return LoadTestResult(
                total_requests=0,
                requests_per_second=0.0,
                response_time_avg=0.0,
                response_time_95p=0.0,
                response_time_99p=0.0,
                failures=0,
                stdout="",
                stderr=str(e),
                exit_code=1
            )

    def _parse_load_output(self, sandbox_result: Dict) -> LoadTestResult:
        """Parse Locust output from sandbox execution."""
        stdout = sandbox_result.get("stdout", "")
        stderr = sandbox_result.get("stderr", "")
        exit_code = sandbox_result.get("exit_code", 1)

        # Default values
        total_requests = 0
        rps = 0.0
        avg_response = 0.0
        p95 = 0.0
        p99 = 0.0
        failures = 0

        # Parse Locust output
        for line in stdout.split('\n'):
            if 'Requests/sec:' in line:
                try:
                    rps = float(line.split('Requests/sec:')[1].strip())
                except:
                    pass
            elif 'Total Requests:' in line:
                try:
                    total_requests = int(line.split('Total Requests:')[1].strip())
                except:
                    pass
            elif 'Average response time:' in line:
                try:
                    avg_response = float(line.split('Average response time:')[1].split()[0])
                except:
                    pass
            elif '95%ile:' in line:
                try:
                    p95 = float(line.split('95%ile:')[1].split()[0])
                except:
                    pass
            elif '99%ile:' in line:
                try:
                    p99 = float(line.split('99%ile:')[1].split()[0])
                except:
                    pass
            elif 'Total Failures:' in line:
                try:
                    failures = int(line.split('Total Failures:')[1].strip())
                except:
                    pass

        return LoadTestResult(
            total_requests=total_requests,
            requests_per_second=rps,
            response_time_avg=avg_response,
            response_time_95p=p95,
            response_time_99p=p99,
            failures=failures,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )