import json
import os
import requests
from providers.base_adapter import BaseAdapter, ModelResponse
from providers.model_registry import choose_model_for_role

class NIMAdapter(BaseAdapter):
    def __init__(self, role: str = "default"):
        if role == "thinkers":
            model = "nvidia/llama-3.1-nemotron-70b-instruct"  # Advanced reasoning model
        elif role == "gateways":
            model = "nvidia/llama-3.1-nemotron-51b-instruct"  # Efficient for routing
        else:
            model = "nvidia/llama-3.1-nemotron-51b-instruct"  # Default
        # Use centralized registry to pick the model for the role
        model = choose_model_for_role(role)
        super().__init__("NVIDIA_NIM_API_KEY", "https://integrate.api.nvidia.com/v1/chat/completions", role=role)
        # Fallback to legacy env var if the expected one isn't present
        if not getattr(self, "api_key", None):
            legacy = os.getenv("NVIDIA_API_KEY")
            if legacy:
                self.api_key = legacy
        self.default_model = model

    def _call_api(self, messages, **kwargs):
        print(f"NIMAdapter _call_api called with messages: {messages}")  # Debug log
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.default_model,
            "messages": messages,
            **kwargs
        }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        print("Returning real API response")  # Debug log
        return response.json()

    def _normalize_response(self, raw):
        choice = raw["choices"][0]
        message = choice["message"]
        text = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        usage = raw.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        try:
            structured = json.loads(text)
        except Exception:
            structured = {}
        confidence = 1.0  # placeholder
        return ModelResponse(text=text, tokens=tokens, tool_calls=tool_calls, structured_response=structured, confidence=confidence, latency_ms=0)

    def _estimate_cost(self, tokens):
        # NVIDIA NIM pricing approximation
        return tokens * 0.000005