import json
import os
import requests
from .base_adapter import BaseAdapter, ModelResponse
from .model_registry import choose_model_for_role

class OllamaAdapter(BaseAdapter):
    def __init__(self, model_name: str = "llama2", role: str = "default"):
        # Ollama typically runs on localhost:11434
        super().__init__("", "http://localhost:11434/api/generate", role=role)
        self.model_name = model_name

    def call_model(self, messages: list, **kwargs) -> ModelResponse:
        # Convert OpenAI-style messages to Ollama format
        prompt = self._messages_to_prompt(messages)

        payload = {
            "model": kwargs.get("model", self.model_name),
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            text = data.get("response", "")
            tokens = len(text.split())  # Rough estimate

            return ModelResponse(
                text=text,
                tokens=tokens,
                tool_calls=[],
                structured_response={},
                confidence=0.8,
                latency_ms=0
            )
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def _messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to a simple prompt for Ollama"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.insert(0, f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)

    def _call_api(self, messages, **kwargs):
        # Not used - call_model handles the logic
        raise NotImplementedError("Use call_model instead")

    def _normalize_response(self, raw_response):
        # Not used - call_model handles the logic
        raise NotImplementedError("Use call_model instead")

    def _estimate_cost(self, tokens):
        # Ollama is free (local)
        return 0.0