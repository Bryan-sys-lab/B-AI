#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_nvidia_models_for_tools():
    """Check which NVIDIA models support tools"""

    api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("No NVIDIA API key found")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # Use same SSL setting as adapter
        verify_ssl = not os.getenv("DISABLE_SSL_VERIFICATION", "").lower() in ("true", "1", "yes")
        response = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers, verify=verify_ssl, timeout=10)
        response.raise_for_status()
        data = response.json()

        print(f"Found {len(data.get('data', []))} models")
        print("\nModels supporting tools:")

        tool_models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            # Check if model supports tools by looking at capabilities or trying a test call
            # For now, just list all models and we'll test a few
            if "tool" in model_id.lower() or "function" in model_id.lower():
                tool_models.append(model_id)
                print(f"  - {model_id}")

        if not tool_models:
            print("  No models found with 'tool' or 'function' in name")

        print("\nAll available models:")
        for model in data.get("data", [])[:20]:  # Show first 20
            model_id = model.get("id", "")
            print(f"  - {model_id}")

        if len(data.get("data", [])) > 20:
            print(f"  ... and {len(data.get('data', [])) - 20} more")

    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    check_nvidia_models_for_tools()