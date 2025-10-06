#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

from providers.nim_adapter import NIMAdapter

def test_nim_role(role):
    try:
        adapter = NIMAdapter(role=role)
        print(f"Role '{role}': Model = {adapter.default_model}")
        return True
    except Exception as e:
        print(f"Role '{role}' failed: {e}")
        return False

def main():
    roles = ["thinkers", "builders", "fix_implementation", "default"]
    print("Testing NIM adapter with different roles:")
    for role in roles:
        test_nim_role(role)

if __name__ == "__main__":
    main()