import json
import requests
from typing import List, Dict
from providers.base_adapter import BaseAdapter, ModelResponse

class DeepSeekAdapter(BaseAdapter):
    def __init__(self, role: str = "default"):
        super().__init__("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1/chat/completions", role=role)
        self.hf_api_key = self._get_secret("HF_API_KEY") if role == "builders" else None

    def _call_api(self, messages, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": kwargs.get("model", "deepseek-chat"),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def call_model(self, messages: List[Dict], **kwargs) -> ModelResponse:
        try:
            return super().call_model(messages, **kwargs)
        except Exception as e:
            if self.role == "builders" and self.hf_api_key:
                self.logger.warning("DeepSeek failed, falling back to HF", error=str(e))
                return self._call_hf_fallback(messages, **kwargs)
            else:
                raise e

    def _call_hf_fallback(self, messages: List[Dict], **kwargs) -> ModelResponse:
        # Simple HF inference API call
        hf_endpoint = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {"inputs": messages, "parameters": {"temperature": kwargs.get("temperature", 0.7)}}
        response = requests.post(hf_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        raw = response.json()
        # Normalize HF response to ModelResponse
        text = raw[0].get("generated_text", "") if raw else ""
        return ModelResponse(text=text, tokens=len(text.split()), tool_calls=[], structured_response={}, confidence=0.5, latency_ms=0)

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
        # DeepSeek pricing approximation
        return tokens * 0.000001