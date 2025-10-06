#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import json
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def test_agentic_tools():
    """Test the full agentic system with natural language tool orchestration"""

    # Test task that should trigger tool usage
    test_request = {
        "description": "Create a simple Python script that reads a file and runs tests on it. First, I need to check what files exist in a repository, then read one, and finally run some tests.",
        "conversation_history": []
    }

    print("Testing agentic system with tool orchestration...")
    print(f"Task: {test_request['description']}")

    try:
        # Call the fix implementation agent (port 8004 from docker-compose)
        response = requests.post("http://localhost:8004/execute", json=test_request, timeout=120)
        response.raise_for_status()
        result = response.json()

        print("✓ SUCCESS - Agentic system responded!")
        print(f"Result: {result.get('result', '')[:500]}...")
        print(f"Success: {result.get('success', False)}")
        print(f"Tokens: {result.get('tokens', 0)}")

        # Check if artifacts were saved
        artifacts = result.get('artifacts_saved', [])
        if artifacts:
            print(f"Artifacts saved: {artifacts}")
        else:
            print("No artifacts saved")

    except Exception as e:
        print(f"✗ FAILED - Agentic system error: {str(e)}")
        if hasattr(e, 'response') and e.response:
            try:
                error_details = e.response.json()
                print(f"Error response: {json.dumps(error_details, indent=2)}")
            except:
                print(f"Error response text: {e.response.text}")

if __name__ == "__main__":
    test_agentic_tools()