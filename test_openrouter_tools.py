#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_openrouter_api_key():
    """Test if OpenRouter API key works"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False

    print(f"🔑 Testing OpenRouter API key: {api_key[:10]}...")

    try:
        # Test with a simple chat completion
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": "Hello, test message"}],
            "max_tokens": 50
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            print("✅ OpenRouter API key is valid")
            return True
        else:
            print(f"❌ OpenRouter API key invalid: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing OpenRouter API key: {e}")
        return False

def get_openrouter_models():
    """Fetch available models from OpenRouter"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found")
        return []

    try:
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            print(f"❌ Failed to fetch models: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        return []

def check_tool_support():
    """Check which models support tool calling"""
    print("\n🔧 Checking tool support for OpenRouter models...")

    models = get_openrouter_models()
    if not models:
        print("❌ No models found")
        return

    tool_supported_models = []
    total_models = len(models)

    for model in models:
        model_id = model.get("id", "")
        capabilities = model.get("capabilities", {})

        # Check if model supports tools
        if capabilities.get("tools", False):
            tool_supported_models.append({
                "id": model_id,
                "name": model.get("name", ""),
                "description": model.get("description", ""),
                "pricing": model.get("pricing", {})
            })

    print(f"📊 Total models: {total_models}")
    print(f"🛠️  Models with tool support: {len(tool_supported_models)}")

    if tool_supported_models:
        print("\n✅ Models that support tool calling:")
        for model in tool_supported_models[:10]:  # Show first 10
            print(f"  - {model['id']} ({model['name']})")
        if len(tool_supported_models) > 10:
            print(f"  ... and {len(tool_supported_models) - 10} more")
    else:
        print("❌ No models found with tool support")

    return tool_supported_models

def test_tool_call():
    """Test actual tool calling with a supported model"""
    tool_models = check_tool_support()
    if not tool_models:
        print("❌ No tool-capable models found, skipping tool call test")
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    model_id = tool_models[0]["id"]  # Use first available tool model

    print(f"\n🧪 Testing tool calling with model: {model_id}")

    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Sample tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "What's the weather in New York?"}
            ],
            "tools": tools,
            "max_tokens": 200
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                print("✅ SUCCESS - Model made tool calls!")
                print(f"Tool calls: {json.dumps(tool_calls, indent=2)}")
            else:
                print("⚠️  Model responded but did not make tool calls")
                print(f"Response: {message.get('content', '')[:200]}...")
        else:
            print(f"❌ Tool call test failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Error testing tool calls: {e}")

if __name__ == "__main__":
    print("🚀 Testing OpenRouter API and Tool Support\n")

    if test_openrouter_api_key():
        check_tool_support()
        test_tool_call()
    else:
        print("❌ Cannot proceed without valid API key")