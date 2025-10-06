#!/usr/bin/env python3

import os
import sys
import json

# Add the server directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from providers.nim_adapter import NIMAdapter

def test_nim_error():
    """Test NIM adapter with a problematic response"""

    # Create adapter
    adapter = NIMAdapter(role="builders")

    # Test message that might trigger the error
    messages = [
        {"role": "user", "content": "make a python function to represent bayes naive function"}
    ]

    try:
        response = adapter.call_model(messages, temperature=0.7)
        print(f"Response text: '{response.text}'")
        print(f"Structured response: {response.structured_response}")
        print(f"Error: {response.error}")
    except Exception as e:
        print(f"Exception raised: {e}")

if __name__ == "__main__":
    test_nim_error()