import json
import os
import requests
from .base_adapter import BaseAdapter, ModelResponse

class TogetherAdapter(BaseAdapter):
    def __init__(self, role: str = "default"):
        super().__init__("TOGETHER_API_KEY", "https://api.together.xyz/v1/chat/completions", role=role)

    def _call_api(self, messages, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": kwargs.get("model", "meta-llama/llama-3-8b-chat-hf"),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]

        response = requests.post(self.endpoint, headers=headers, json=payload, verify=not os.getenv("DISABLE_SSL_VERIFICATION", "").lower() in ("true", "1", "yes"))
        response.raise_for_status()
        return response.json()

    def _normalize_response(self, raw_response):
        choices = raw_response.get("choices", [])
        if not choices:
            return ModelResponse(text="", tokens=0, tool_calls=[], structured_response={}, confidence=0.0, latency_ms=0)
        choice = choices[0]
        message = choice["message"]
        text = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        usage = raw_response.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        try:
            structured = json.loads(text)
        except Exception:
            structured = {}
        confidence = 1.0
        return ModelResponse(text=text, tokens=tokens, tool_calls=tool_calls, structured_response=structured, confidence=confidence, latency_ms=0)

    def _estimate_cost(self, tokens):
        return tokens * 0.0000008  # Together pricing approximation