import os
import sys
from typing import List
import re

# Ensure repo root is on sys.path so imports like `providers.*` work when the
# service is packaged or run from a subdirectory. Calculate repo root from
# this file's location.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    # Intentional import-time sys.path modification to ensure repo root is
    # available in containerized or subdirectory run contexts. Lint rule E402
    # (module level import not at top) is intentionally suppressed here.
    sys.path.insert(0, _repo_root)  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402

class PatchGenerator:
    def __init__(self):
        self.adapter = NIMAdapter(role="builders")

    def generate_patches(self, prompt: str) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        response = self.adapter.call_model(messages)

        # Parse the response for unified diffs
        diffs = self._extract_diffs(response.text)
        return diffs

    def _extract_diffs(self, text: str) -> List[str]:
        # Simple regex to find unified diff blocks
        diff_pattern = r'(---.*?\n\+\+\+.*?\n@@.*?@@.*?(?=\n---|\Z))'
        matches = re.findall(diff_pattern, text, re.DOTALL)
        return matches

    def _fallback_to_hf(self, prompt: str) -> List[str]:
        # Placeholder for local HF fallback
        # For MVP, return empty
        return []