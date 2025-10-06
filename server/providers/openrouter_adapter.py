from typing import List, Dict

from .base_adapter import BaseAdapter, ModelResponse
from .nim_adapter import NIMAdapter

class OpenRouterAdapter(BaseAdapter):
    def __init__(self, role: str = "default"):
        # Stub: no longer needs API key or endpoint, delegates to NIM
        super().__init__("", "", role=role)
        self.nim_adapter = NIMAdapter(role=role)

    def call_model(self, messages: List[Dict], **kwargs) -> ModelResponse:
        # Stub: delegate all calls to NIMAdapter
        return self.nim_adapter.call_model(messages, **kwargs)

    def _call_api(self, messages, **kwargs):
        # Stub: not used
        raise NotImplementedError("OpenRouterAdapter is stubbed and delegates to NIMAdapter")

    def _normalize_response(self, raw):
        # Stub: not used
        raise NotImplementedError("OpenRouterAdapter is stubbed and delegates to NIMAdapter")

    def _estimate_cost(self, tokens):
        # Delegate to NIM pricing
        return self.nim_adapter._estimate_cost(tokens)