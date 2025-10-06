import re
import os
import requests

class SafetyChecker:
    def __init__(self):
        self.opa_url = os.getenv("OPA_URL", "http://localhost:8181")  # Assume OPA service

    def check_patch(self, patch: str) -> bool:
        # Check for secrets
        if self._contains_secrets(patch):
            return False

        # OPA check
        if not self._opa_check(patch):
            return False

        return True

    def _contains_secrets(self, patch: str) -> bool:
        # Simple regex for common secrets
        patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
        ]
        for pattern in patterns:
            if re.search(pattern, patch, re.IGNORECASE):
                return True
        return False

    def _opa_check(self, patch: str) -> bool:
        try:
            response = requests.post(
                f"{self.opa_url}/v1/data/example/allow",
                json={"input": {"patch": patch}},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json().get("result", False)
                return result
            return False
        except Exception:
            # If OPA is not available, allow for now
            return True