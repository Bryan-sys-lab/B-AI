#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import json
from dotenv import load_dotenv
from agents.fix_implementation.tool_orchestrator import ToolOrchestrator

# Load environment variables from .env file
load_dotenv()

def test_tool_orchestrator():
    """Test the tool orchestrator with sample responses"""

    orchestrator = ToolOrchestrator()

    # Test 1: Parse function calls
    response_with_functions = """
    I need to analyze this codebase. Let me start by reading the main file.

    I should run: git_read_file(repo_url="https://github.com/example/repo", file_path="main.py")

    Then I'll check the tests: run_tests(repo_url="https://github.com/example/repo", test_command="pytest")
    """

    print("Testing tool parsing...")
    tool_requests = orchestrator.parse_tool_requests(response_with_functions)
    print(f"Found {len(tool_requests)} tool requests:")
    for req in tool_requests:
        print(f"  - {req['function']}: {req['args']}")

    # Test 2: Format tool results
    sample_results = [
        {
            'request': {'function': 'git_read_file', 'args': {'repo_url': 'https://github.com/example/repo', 'file_path': 'main.py'}},
            'result': {'content': 'print("Hello World")', 'success': True}
        },
        {
            'request': {'function': 'run_tests', 'args': {'repo_url': 'https://github.com/example/repo', 'test_command': 'pytest'}},
            'result': {'output': '2 passed, 0 failed', 'success': True}
        }
    ]

    print("\nTesting result formatting...")
    formatted = orchestrator.format_tool_results(sample_results)
    print("Formatted results:")
    print(formatted)

    print("\nTool orchestrator test completed!")

if __name__ == "__main__":
    test_tool_orchestrator()