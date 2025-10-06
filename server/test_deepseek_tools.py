#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import json
from dotenv import load_dotenv
from providers.deepseek_adapter import DeepSeekAdapter

# Load environment variables from .env file
load_dotenv()

def test_deepseek_tools():
    """Test if DeepSeek supports tools"""

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

    print("Testing DeepSeek tools support...")
    print(f"Tools: {json.dumps(tools, indent=2)}")

    try:
        adapter = DeepSeekAdapter(role="builders")

        response = adapter.call_model(messages, temperature=0.7, tools=tools)
        print("✓ SUCCESS - DeepSeek supports tools!")
        print(f"Response: {response.text[:200]}...")
        print(f"Tool calls: {response.tool_calls}")

    except Exception as e:
        print(f"✗ FAILED - DeepSeek tools error: {str(e)}")
        print(f"Exception type: {type(e)}")

        # Try to get more details if it's an HTTP error
        if hasattr(e, 'response') and e.response:
            try:
                error_details = e.response.json()
                print(f"Error response: {json.dumps(error_details, indent=2)}")
            except:
                print(f"Error response text: {e.response.text}")

if __name__ == "__main__":
    test_deepseek_tools()