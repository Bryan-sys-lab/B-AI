import json
from providers.base_adapter import BaseAdapter, ModelResponse

class MistralAdapter(BaseAdapter):
    def __init__(self):
        super().__init__("MISTRAL_API_KEY", "https://api.mistral.ai/v1/chat/completions")

    def _call_api(self, messages, **kwargs):
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": kwargs.get("model", "mistral-medium"),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
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
        # Mistral pricing approximation
        return tokens * 0.000002