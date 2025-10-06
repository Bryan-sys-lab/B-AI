from providers.nim_adapter import NIMAdapter
from typing import List, Dict, Any

class FailureAnalyzer:
    def __init__(self, provider: str = "nim"):
        self.adapter = NIMAdapter()

    def analyze_failures(self, test_output: str, errors: List[str]) -> str:
        errors_str = "\n".join(errors)
        prompt = f"""
Analyze the following test output and errors to identify the root cause of the failures.

Test Output:
{test_output}

Errors:
{errors_str}

Provide a detailed analysis of the root cause, including:
1. What is failing
2. Why it's failing
3. Potential fixes
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.adapter.call_model(messages, temperature=0.2)
        return response.text