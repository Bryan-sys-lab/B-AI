#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import json
from providers.nim_adapter import NIMAdapter

def test_tools_error():
    """Test what error occurs when tools are included"""

    # Simple message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what can you do?"}
    ]

    # Sample tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    print("Testing tools error with NVIDIA NIM...")
    print(f"Tools: {json.dumps(tools, indent=2)}")

    try:
        adapter = NIMAdapter(role="builders")
        # Force enable tools temporarily
        adapter._model_supports_tools = lambda model_name: True

        response = adapter.call_model(messages, temperature=0.7, tools=tools)
        print("✓ SUCCESS - Tools worked!")
        print(f"Response: {response.text[:200]}...")

    except Exception as e:
        print(f"✗ FAILED - Tools error: {str(e)}")
        print(f"Exception type: {type(e)}")

        # Try to get more details if it's an HTTP error
        if hasattr(e, 'response') and e.response:
            try:
                error_details = e.response.json()
                print(f"Error response: {json.dumps(error_details, indent=2)}")
            except:
                print(f"Error response text: {e.response.text}")

if __name__ == "__main__":
    test_tools_error()