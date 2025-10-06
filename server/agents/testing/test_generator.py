import logging
from typing import List, Dict, Optional
from providers.nim_adapter import NIMAdapter

logger = logging.getLogger(__name__)

class TestGenerator:
    def __init__(self):
        self.adapter = NIMAdapter(role="builders")

    def generate_unit_tests(self, code: str, language: str = "python") -> List[str]:
        """Generate unit tests for the given code."""
        prompt = f"""
        Generate comprehensive unit tests for the following {language} code.
        Focus on edge cases, error handling, and normal operation.
        Return only the test code, no explanations.

        Code to test:
        {code}
        """
        return self._generate_tests(prompt, "unit")

    def generate_integration_tests(self, code: str, language: str = "python") -> List[str]:
        """Generate integration tests for the given code."""
        prompt = f"""
        Generate integration tests for the following {language} code.
        Test interactions between components, data flow, and system behavior.
        Return only the test code, no explanations.

        Code to test:
        {code}
        """
        return self._generate_tests(prompt, "integration")

    def generate_fuzz_tests(self, code: str, language: str = "python") -> List[str]:
        """Generate fuzz tests for the given code."""
        prompt = f"""
        Generate fuzz tests for the following {language} code.
        Focus on random inputs, boundary conditions, and potential crashes.
        Use appropriate fuzzing libraries for {language}.
        Return only the test code, no explanations.

        Code to test:
        {code}
        """
        return self._generate_tests(prompt, "fuzz")

    def generate_performance_tests(self, code: str, language: str = "python") -> List[str]:
        """Generate performance tests for the given code."""
        prompt = f"""
        Generate performance tests for the following {language} code.
        Include benchmarks, load tests, and memory usage tests.
        Return only the test code, no explanations.

        Code to test:
        {code}
        """
        return self._generate_tests(prompt, "performance")

    def generate_ui_tests(self, code: str, language: str = "python") -> List[str]:
        """Generate UI tests for the given code (assuming web app)."""
        prompt = f"""
        Generate Selenium-based UI tests for the following {language} web application code.
        Test user interactions, page loads, and UI components.
        Return only the test code, no explanations.

        Code to test:
        {code}
        """
        return self._generate_tests(prompt, "ui")

    def generate_load_tests(self, code: str, language: str = "python") -> List[str]:
        """Generate Locust-based load tests for the given code."""
        prompt = f"""
        Generate Locust load tests for the following {language} application code.
        Test concurrent users, response times, and system load.
        Return only the test code, no explanations.

        Code to test:
        {code}
        """
        return self._generate_tests(prompt, "load")

    def _generate_tests(self, prompt: str, test_type: str) -> List[str]:
        """Internal method to generate tests using DeepSeek."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.adapter.call_model(messages, temperature=0.3)
            test_code = response.text.strip()

            # Split into individual test functions/methods if multiple
            tests = [test.strip() for test in test_code.split('\n\n') if test.strip()]
            logger.info(f"Generated {len(tests)} {test_type} tests")
            return tests
        except Exception as e:
            logger.error(f"Error generating {test_type} tests: {str(e)}")
            return []