#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

from providers.nim_adapter import NIMAdapter

def test_nim():
    try:
        # Test different roles/models
        roles = ["default", "builders", "thinkers"]
        for role in roles:
            print(f"\nTesting role: {role}")
            adapter = NIMAdapter(role=role)
            print(f"Adapter initialized with model: {adapter.default_model}")

            # Test a simple call
            messages = [{"role": "user", "content": "Hello, can you tell me what 2+2 equals?"}]
            response = adapter.call_model(messages)
            print(f"Response: {response.text[:100]}...")
            print(f"Tokens used: {response.tokens}")
            print(f"Success for {role}")
    except Exception as e:
        print(f"NIM adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("All NIM adapter tests successful!")
    return True

if __name__ == "__main__":
    test_nim()