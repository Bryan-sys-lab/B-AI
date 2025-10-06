from typing import List, Dict

from .base_adapter import BaseAdapter, ModelResponse
from .nim_adapter import NIMAdapter

class MistralAdapter(BaseAdapter):
    def __init__(self, role: str = "default"):
        # Stub: delegates to NIM
        super().__init__("MISTRAL_API_KEY", "", role=role)
        self.nim_adapter = NIMAdapter(role=role)

    def call_model(self, messages: List[Dict], **kwargs) -> ModelResponse:
        # Delegate all calls to NIMAdapter
        return self.nim_adapter.call_model(messages, **kwargs)

    def _call_api(self, messages, **kwargs):
        # Delegate to NIMAdapter
        return self.nim_adapter._call_api(messages, **kwargs)

    def _normalize_response(self, raw_response):
        # Delegate to NIMAdapter
        return self.nim_adapter._normalize_response(raw_response)

    def _estimate_cost(self, tokens):
        # Delegate to NIM pricing
        return self.nim_adapter._estimate_cost(tokens)