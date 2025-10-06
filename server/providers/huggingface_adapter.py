from typing import List, Dict, Any

from .base_adapter import BaseAdapter, ModelResponse
from .nim_adapter import NIMAdapter

class HuggingFaceAdapter(BaseAdapter):
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", role: str = "default"):
        # Stub: no longer loads local models, delegates to NIM
        super().__init__("", "", role=role)
        self.nim_adapter = NIMAdapter(role=role)

    def call_model(self, messages: List[Dict], **kwargs) -> ModelResponse:
        # Stub: delegate all calls to NIMAdapter
        return self.nim_adapter.call_model(messages, **kwargs)

    def _call_api(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        # Stub: not used
        raise NotImplementedError("HuggingFaceAdapter is stubbed and delegates to NIMAdapter")

    def _normalize_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        # Stub: not used
        raise NotImplementedError("HuggingFaceAdapter is stubbed and delegates to NIMAdapter")

    def _estimate_cost(self, tokens: int) -> float:
        # Delegate to NIM pricing
        return self.nim_adapter._estimate_cost(tokens)