"""Model registry and role mapping for provider adapters.

This module centralizes which models to use for each role and provides
fallback behavior. Adapters should consult this registry to choose the
appropriate model for a call.
"""
import os
import re
import requests
from typing import Dict, List, Any

def get_available_models():
    """Fetch available models from NVIDIA API."""
    api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        # Use same SSL verification setting as NIMAdapter
        verify_ssl = not os.getenv("DISABLE_SSL_VERIFICATION", "").lower() in ("true", "1", "yes")
        response = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers, verify=verify_ssl, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [model["id"] for model in data.get("data", [])]
    except Exception:
        return []


ROLE_MAPPING: Dict[str, Any] = {
    "thinkers": {
        "description": "Reasoning and planning models",
        "preferred_patterns": [
            r"meta/llama.*70b.*instruct",
            r"meta/llama.*405b.*instruct",
            r"mistralai/mistral-large",
            r"deepseek-ai/deepseek.*instruct"
        ],
        "fallback_patterns": [
            r"meta/llama.*instruct",
            r"mistralai/.*instruct",
            r".*instruct"
        ]
    },
    "builders": {
        "description": "Code generation models",
        "preferred_patterns": [
            r"meta/codellama.*instruct",
            r"deepseek-ai/deepseek.*coder.*instruct",
            r"mistralai/codestral.*instruct",
            r"qwen/qwen.*coder.*instruct"
        ],
        "fallback_patterns": [
            r"meta/llama.*instruct",
            r".*coder.*instruct",
            r".*instruct"
        ]
    },
    "fix_implementation": {
        "description": "Code repair and implementation",
        "preferred_patterns": [
            r"meta/codellama.*instruct",
            r"deepseek-ai/deepseek.*coder.*instruct",
            r"mistralai/codestral.*instruct"
        ],
        "fallback_patterns": [
            r"meta/llama.*instruct",
            r".*coder.*instruct",
            r".*instruct"
        ]
    },
    "task_classifier": {
        "description": "Intelligent task classification and routing",
        "preferred_patterns": [
            r"meta/llama.*70b.*instruct",
            r"meta/llama.*405b.*instruct",
            r"mistralai/mistral-large",
            r"deepseek-ai/deepseek.*instruct"
        ],
        "fallback_patterns": [
            r"meta/llama.*instruct",
            r"mistralai/.*instruct",
            r".*instruct"
        ]
    },
    "default": {
        "description": "General-purpose fallback",
        "preferred_patterns": [
            r"meta/llama.*8b.*instruct",
            r"meta/llama.*3b.*instruct",
            r"mistralai/mistral.*instruct"
        ],
        "fallback_patterns": [
            r".*instruct",
            r".*chat"
        ]
    },
    "gateways": {
        "description": "Unified API and routing layer",
        "primary": "NVIDIA NIM Gateway",
        "alternative": "NVIDIA Triton Inference Server"
    },
    "fallbacks": {
        "description": "Offline / On-premise safety net",
        "providers": [
            "Self-hosted NVIDIA NIM (Nemotron, CodeLlama)",
            "NVIDIA Triton Inference Server",
            "NVIDIA NeMo Guardrails"
        ]
    }
}


def validate_model_availability(model_name: str) -> bool:
    """Test if a model is actually available for use by making a minimal API call."""
    api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1  # Minimal response to test availability
    }

    try:
        # Use same SSL verification setting as NIMAdapter
        verify_ssl = not os.getenv("DISABLE_SSL_VERIFICATION", "").lower() in ("true", "1", "yes")
        response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions",
                               headers=headers, json=payload, verify=verify_ssl, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def choose_model_for_role(role: str) -> str:
    """Return the preferred model name for a given role using pattern matching.

    Dynamically selects from available models by matching against preferred patterns,
    then fallback patterns, prioritizing larger/better models and validating availability.
    """
    available_models = get_available_models()

    # If no models are available at all, return a default
    if not available_models:
        return "mistralai/mistral-7b-instruct-v0.3"

    role_cfg = ROLE_MAPPING.get(role, ROLE_MAPPING.get("default", {}))
    if not role_cfg or "preferred_patterns" not in role_cfg:
        # Fallback for roles without patterns or default role without patterns
        instruct_models = [m for m in available_models if 'instruct' in m.lower()]
        if instruct_models:
            # Validate and return first working model
            for model in instruct_models:
                if validate_model_availability(model):
                    return model
            # If none work, return first one anyway
            return instruct_models[0]
        return available_models[0]

    # Helper function to find matching models
    def find_matching_models(patterns):
        matches = []
        for pattern in patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for model in available_models:
                    if regex.search(model):
                        matches.append(model)
            except re.error:
                # Skip invalid regex patterns
                continue
        return matches

    # Try preferred patterns first
    preferred_matches = find_matching_models(role_cfg.get("preferred_patterns", []))
    if preferred_matches:
        # Sort by model size/capability (rough heuristic: prefer larger numbers, longer names)
        preferred_matches.sort(key=lambda x: (len(x), sum(int(num) for num in re.findall(r'\d+', x))), reverse=True)
        # Validate each model and return the first working one
        for model in preferred_matches:
            if validate_model_availability(model):
                return model

    # Try fallback patterns
    fallback_matches = find_matching_models(role_cfg.get("fallback_patterns", []))
    if fallback_matches:
        # Sort by model size/capability
        fallback_matches.sort(key=lambda x: (len(x), sum(int(num) for num in re.findall(r'\d+', x))), reverse=True)
        # Validate each model and return the first working one
        for model in fallback_matches:
            if validate_model_availability(model):
                return model

    # Final fallback: any available instruct model
    instruct_models = [m for m in available_models if 'instruct' in m.lower()]
    for model in instruct_models:
        if validate_model_availability(model):
            return model

    # Last resort: any available model (even if not validated)
    return available_models[0]
