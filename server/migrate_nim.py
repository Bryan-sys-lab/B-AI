import json

# Ranked NVIDIA NIM roles
NIM_MODEL_MAP = {
    "thinkers": {
        "description": "Reasoning and planning models",
        "primary": "nvidia/nemotron-4-340b-instruct",
        "secondary": "nvidia/llama-3.1-nemotron-70b-instruct",
        "fallback": ["nvidia/nemotron-4-15b-instruct", "nvidia/llama-3.1-nemotron-51b-instruct", "mistralai/mistral-large", "mistralai/mistral-7b-instruct", "deepseek-chat", "llama2", "codellama"]
    },
    "builders": {
        "description": "Code generation models",
        "primary": "nvidia/usdcode-llama-3.1-70b-instruct",
        "secondary": "nvidia/llama-3.1-nemotron-70b-instruct",
        "fallback": ["nvidia/llama-3.1-nemotron-51b-instruct", "nvidia/llama3-chatqa-1.5-8b", "deepseek-coder", "codellama", "llama2", "mistralai/codestral-22b-instruct-v0.1", "qwen/qwen2.5-coder-32b-instruct"]
    },
    "fix_implementation": {
        "description": "Code repair and implementation",
        "primary": "nvidia/usdcode-llama-3.1-70b-instruct",
        "secondary": "nvidia/llama-3.1-nemotron-70b-instruct",
        "fallback": ["nvidia/llama-3.1-nemotron-51b-instruct", "nvidia/llama3-chatqa-1.5-8b", "codellama", "deepseek-coder", "llama2", "mistralai/codestral-22b-instruct-v0.1"]
    },
    "default": {
        "description": "General-purpose fallback",
        "primary": "nvidia/nemotron-4-15b-instruct",
        "secondary": "mistralai/mistral-large",
        "fallback": ["nvidia/llama-3.1-nemotron-51b-instruct", "codellama"]
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

PROVIDER_ADAPTER_PRINCIPLES = {
    "normalize_api": "All adapters must return standardized ModelResponse objects: {text, tokens, tool_calls, structured_response, confidence, latency_ms}",
    "retries_and_backoff": "Adapters implement retry logic with exponential backoff and emit metrics on retries/failures.",
    "metrics": {
        "per_call": ["latency_ms", "cost_estimate", "tokens_used", "success_rate"],
        "export": ["Prometheus", "Grafana dashboards"]
    },
    "function_call_schema": "Adapters must support function-calling / tool invocation emulation with a standard JSON schema.",
    "observability": {
        "tracing": "OpenTelemetry traces emitted per adapter call",
        "logging": "Structured logs with request_id, provider, latency, error codes"
    },
    "security": {
        "secret_handling": "Never expose API keys in logs or prompts",
        "opa_policy_checks": "All outputs validated against Open Policy Agent before commit or deployment"
    }
}

def create_nim_config(new_config_path: str):
    """Create a new NIM-only config with ranked roles."""
    print("Creating new config with NIM mappings")
    new_config = {
        "role_mapping_without_openrouter": NIM_MODEL_MAP,
        "provider_adapter_principles": PROVIDER_ADAPTER_PRINCIPLES
    }
    print(f"New config structure: {new_config.keys()}")

    print(f"Saving new config to {new_config_path}")
    try:
        with open(new_config_path, "w") as f:
            json.dump(new_config, f, indent=2)
        print(f"✅ Config creation complete. New config saved at {new_config_path}")
    except Exception as e:
        print(f"❌ Failed to save new config: {e}")


# Example usage
if __name__ == "__main__":
    create_nim_config("new_nim_config.json")