from typing import List, Optional

class PromptBuilder:
    def build_prompt(self, failing_tests: List[str], context: Optional[str] = None) -> str:
        prompt = "You are a code generation assistant. Your task is to generate code patches to fix failing tests.\n\n"

        if context:
            prompt += f"Context:\n{context}\n\n"

        prompt += "Failing tests:\n"
        for test in failing_tests:
            prompt += f"- {test}\n"

        prompt += "\nPlease generate a unified diff patch that fixes these failing tests. The patch should be in unified-diff format.\n"
        prompt += "Output only the diff, no explanations."

        return prompt