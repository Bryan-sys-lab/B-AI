import time
import json
import requests
from pathlib import Path
from datetime import datetime

# ==============================
# Config
# ==============================
INTERVAL_MINUTES = 5  # how often to re-run benchmark

API_KEYS = {
    "openrouter": "YOUR_OPENROUTER_KEY",
    "together": "YOUR_TOGETHER_KEY",
    "huggingface": "YOUR_HF_KEY",
    "scaleway": "YOUR_SCALEWAY_KEY",
    "nvidia_nim": "YOUR_NVIDIA_KEY"
}

PROMPTS = {
    "thinkers": "Plan a 3-step strategy for teaching quantum mechanics to high school students.",
    "builders": "Write a Python function that implements quicksort.",
    "fix_implementation": "Here is buggy code: def add(a, b): return a - b. Fix it.",
    "default": "Tell me a fun fact about space."
}

PROVIDERS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "models": ["meta-llama/llama-3.1-8b-instruct"],
        "headers": lambda key: {"Authorization": f"Bearer {key}"}
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "models": ["meta-llama/llama-3-8b-chat-hf"],
        "headers": lambda key: {"Authorization": f"Bearer {key}"}
    },
    "huggingface": {
        "url": "https://api-inference.huggingface.co/models",
        "models": ["mistralai/Mistral-7B-Instruct-v0.3"],
        "headers": lambda key: {"Authorization": f"Bearer {key}"}
    },
    "scaleway": {
        "url": "https://api.scaleway.ai/v1alpha1/models",
        "models": ["scaleway/llama-3.1-8b-instruct"],
        "headers": lambda key: {"Authorization": f"Bearer {key}"}
    },
    "nvidia_nim": {
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "models": ["meta/llama-3.1-8b-instruct"],
        "headers": lambda key: {"Authorization": f"Bearer {key}"}
    }
}

PROVIDER_ADAPTER_PRINCIPLES = {
    "normalize_api": (
        "All adapters must return standardized ModelResponse objects: "
        "{text, tokens, tool_calls, structured_response, confidence, latency_ms}"
    ),
    "retries_and_backoff": (
        "Adapters implement retry logic with exponential backoff and emit metrics on retries/failures."
    ),
    "metrics": {
        "per_call": ["latency_ms", "cost_estimate", "tokens_used", "success_rate"],
        "export": ["Prometheus", "Grafana dashboards"]
    },
    "function_call_schema": (
        "Adapters must support function-calling / tool invocation emulation with a standard JSON schema."
    ),
    "observability": {
        "tracing": "OpenTelemetry traces emitted per adapter call",
        "logging": "Structured logs with request_id, provider, latency, error codes"
    },
    "security": {
        "secret_handling": "Never expose API keys in logs or prompts",
        "opa_policy_checks": (
            "All outputs validated against Open Policy Agent before commit or deployment"
        )
    }
}


def log(msg: str):
    """Simple logger with timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def run_request(provider, cfg, model, prompt, key):
    headers = cfg["headers"](key)

    if provider == "huggingface":
        url = f"{cfg['url']}/{model}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    else:
        url = cfg["url"]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200
        }

    start = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        latency = round((time.time() - start) * 1000, 2)

        if resp.status_code == 200:
            data = resp.json()
            text = ""

            if provider == "huggingface":
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    text = data.get("generated_text", "")
            elif "choices" in data:
                text = data["choices"][0].get("message", {}).get("content", "")

            return {
                "provider": provider,
                "model": model,
                "latency_ms": latency,
                "response_length": len(text),
                "text": text
            }
        else:
            return {
                "provider": provider,
                "model": model,
                "error": resp.text,
                "latency_ms": latency
            }
    except Exception as e:
        return {
            "provider": provider,
            "model": model,
            "error": str(e)
        }


def benchmark():
    all_results = {role: [] for role in PROMPTS}

    for role, prompt in PROMPTS.items():
        log(f"🧪 Testing role: {role}")
        for provider, cfg in PROVIDERS.items():
            key = API_KEYS.get(provider)
            if not key:
                log(f"⚠️ Skipping {provider}, no API key.")
                continue

            for model in cfg["models"]:
                log(f" → {provider}/{model}")
                result = run_request(provider, cfg, model, prompt, key)
                all_results[role].append(result)

    return all_results


def pick_best_models(results):
    role_map = {}

    for role, candidates in results.items():
        valid = [c for c in candidates if "error" not in c]
        if not valid:
            role_map[role] = {"primary": None, "secondary": None, "fallback": []}
            continue

        ranked = sorted(valid, key=lambda x: (x["latency_ms"], -x["response_length"]))

        primary = ranked[0]["model"]
        secondary = ranked[1]["model"] if len(ranked) > 1 else None
        fallback = [r["model"] for r in ranked[2:]] if len(ranked) > 2 else []

        role_map[role] = {
            "description": PROMPTS[role],
            "primary": primary,
            "secondary": secondary,
            "fallback": fallback
        }

    role_map["gateways"] = {
        "description": "Unified API and routing layer",
        "primary": "OpenRouter Gateway",
        "alternative": "NVIDIA Triton Inference Server"
    }
    role_map["fallbacks"] = {
        "description": "Offline / On-prem safety net",
        "providers": [
            "Self-hosted open models (Llama, Mistral, CodeLlama)",
            "NVIDIA Triton Inference Server",
            "NVIDIA NeMo Guardrails"
        ]
    }

    return role_map


def main_loop():
    Path("benchmarks/history").mkdir(parents=True, exist_ok=True)

    while True:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log("🚀 Starting benchmark cycle...")
        results = benchmark()
        best_models = pick_best_models(results)

        # overwrite latest files
        with open("benchmarks/raw_results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open("benchmarks/nim_model_map.json", "w") as f:
            json.dump(best_models, f, indent=2)

        final_config = {
            "role_mapping_nim_only": best_models,
            "provider_adapter_principles": PROVIDER_ADAPTER_PRINCIPLES
        }
        with open("benchmarks/final_config.json", "w") as f:
            json.dump(final_config, f, indent=2)

        # save history snapshot
        with open(f"benchmarks/history/raw_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(f"benchmarks/history/final_config_{timestamp}.json", "w") as f:
            json.dump(final_config, f, indent=2)

        # Auto-update the live registry
        try:
            # Read current registry
            with open("providers/model_registry.py", "r") as f:
                content = f.read()

            # Replace ROLE_MAPPING
            import re
            role_mapping_str = f"ROLE_MAPPING: Dict[str, Any] = {json.dumps(best_models, indent=4)}"
            # Find the ROLE_MAPPING block
            pattern = r'ROLE_MAPPING: Dict\[str, Any\] = \{.*?\n\}'
            new_content = re.sub(pattern, role_mapping_str, content, flags=re.DOTALL)

            with open("providers/model_registry.py", "w") as f:
                f.write(new_content)

            log("✅ Registry auto-updated with new model mappings")

            # Also update a live config file
            with open("live_config.json", "w") as f:
                json.dump(final_config, f, indent=2)

            log("✅ Live config updated")

        except Exception as e:
            log(f"❌ Failed to auto-update: {e}")

        log(f"✅ Benchmark complete! Snapshot saved with timestamp {timestamp}")
        log(f"⏳ Sleeping {INTERVAL_MINUTES} minutes...\n")
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main_loop()