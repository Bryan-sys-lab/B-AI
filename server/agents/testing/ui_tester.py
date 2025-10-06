import os
import logging
from typing import Dict, List, Optional
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class UITestResult(BaseModel):
    passed: int
    failed: int
    errors: int
    total: int
    screenshots: List[str] = []
    stdout: str
    stderr: str
    exit_code: int

class UITester:
    def __init__(self, sandbox_executor_url: str = "http://localhost:8002"):
        self.sandbox_url = sandbox_executor_url

    def run_ui_tests(self, test_code: str, app_url: str = "http://localhost:3000") -> UITestResult:
        """Run Selenium UI tests in sandbox."""
        try:
            # Create test script
            test_script = f"""
import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Setup Chrome options for headless
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

# Create WebDriver
driver = webdriver.Chrome(options=chrome_options)

try:
    # Test code
    {test_code}

    print("UI tests completed successfully")
except Exception as e:
    print(f"UI test error: {{e}}")
    sys.exit(1)
finally:
    driver.quit()
"""

            # Write test script to file and run in sandbox
            payload = {
                "command": "python3",
                "args": ["-c", test_script],
                "working_dir": "/workspace",
                "env": {
                    "DISPLAY": ":99",  # For headless
                    "PYTHONPATH": "/workspace"
                },
                "timeout": 120  # 2 minutes for UI tests
            }

            response = requests.post(f"{self.sandbox_url}/execute", json=payload, timeout=130)
            response.raise_for_status()
            result = response.json()

            # Parse results
            return self._parse_ui_output(result)

        except Exception as e:
            logger.error(f"Error running UI tests: {str(e)}")
            return UITestResult(
                passed=0, failed=0, errors=1, total=0,
                stdout="", stderr=str(e), exit_code=1
            )

    def _parse_ui_output(self, sandbox_result: Dict) -> UITestResult:
        """Parse UI test output from sandbox execution."""
        stdout = sandbox_result.get("stdout", "")
        stderr = sandbox_result.get("stderr", "")
        exit_code = sandbox_result.get("exit_code", 1)

        # Simple parsing - assume tests pass if no errors
        if exit_code == 0:
            passed = 1  # Assume at least one test passed
            failed = 0
            errors = 0
        else:
            passed = 0
            failed = 1
            errors = 1

        total = passed + failed + errors

        return UITestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            total=total,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )