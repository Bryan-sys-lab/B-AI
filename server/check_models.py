import os
import requests
from providers.model_registry import ROLE_MAPPING

def get_available_models():
    api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("No API key found")
        return []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers)
        response.raise_for_status()
        data = response.json()
        models = [model["id"] for model in data.get("data", [])]
        return models
    except Exception as e:
        print(f"Error: {e}")
        return []

available_models = get_available_models()

# Collect all models from ROLE_MAPPING
registry_models = set()
for role, cfg in ROLE_MAPPING.items():
    if "primary" in cfg and cfg["primary"]:
        registry_models.add(cfg["primary"])
    if "secondary" in cfg and cfg["secondary"]:
        registry_models.add(cfg["secondary"])
    if "fallback" in cfg and isinstance(cfg["fallback"], list):
        registry_models.update(cfg["fallback"])

print("Models in registry:")
for model in sorted(registry_models):
    status = "AVAILABLE" if model in available_models else "NOT AVAILABLE"
    print(f"{model}: {status}")

print(f"\nTotal registry models: {len(registry_models)}")
print(f"Available: {len([m for m in registry_models if m in available_models])}")
print(f"Not available: {len([m for m in registry_models if m not in available_models])}")