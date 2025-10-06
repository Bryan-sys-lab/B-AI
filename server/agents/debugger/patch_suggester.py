from providers.nim_adapter import NIMAdapter
from typing import List

class PatchRequest:
    def __init__(self, diff: str, description: str, confidence: float):
        self.diff = diff
        self.description = description
        self.confidence = confidence

class PatchSuggester:
    def __init__(self, provider: str = "nim"):
        self.adapter = NIMAdapter()

    def suggest_patches(self, analysis: str, code_context: str) -> List[PatchRequest]:
        prompt = f"""
Based on the following failure analysis, suggest a code patch to fix the issue.

Analysis:
{analysis}

Code Context:
{code_context}

Provide a diff patch that fixes the problem, along with a description and confidence score (0-1).
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.adapter.call_model(messages, temperature=0.2)
        # Parse response to extract patches
        # For simplicity, assume response contains diff, description, confidence
        # In real implementation, parse structured output
        patches = []
        text = response.text
        # Mock parsing
        if "diff" in text.lower():
            patches.append(PatchRequest(
                diff="mock diff",
                description="Suggested fix based on analysis",
                confidence=0.8
            ))
        return patches