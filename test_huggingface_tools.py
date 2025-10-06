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

def test_huggingface_api_key():
    """Test if HuggingFace API key works"""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY not found in environment")
        return False

    print(f"🔑 Testing HuggingFace API key: {api_key[:10]}...")

    try:
        # Test with a simple inference call
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": "Hello, this is a test message",
            "parameters": {"max_new_tokens": 50}
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            print("✅ HuggingFace API key is valid")
            return True
        else:
            print(f"❌ HuggingFace API key invalid: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing HuggingFace API key: {e}")
        return False

def test_huggingface_chat_api():
    """Test HuggingFace's chat completions API for tool support"""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY not found")
        return

    print("\n💬 Testing HuggingFace Chat Completions API...")

    # Try some models that might support tools
    tool_capable_models = [
        "microsoft/WizardLM-2-8x22B",  # Known to support tools
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-2-7b-chat-hf"
    ]

    for model in tool_capable_models:
        print(f"\n🧪 Testing model: {model}")

        try:
            url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
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
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant with access to tools."},
                    {"role": "user", "content": "What's the weather in New York?"}
                ],
                "tools": tools,
                "max_tokens": 200
            }

            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    tool_calls = message.get("tool_calls", [])

                    if tool_calls:
                        print(f"✅ SUCCESS - {model} supports tool calling!")
                        print(f"Tool calls: {json.dumps(tool_calls, indent=2)}")
                        return True  # Found one that works
                    else:
                        print(f"⚠️  {model} responded but did not use tools")
                        content = message.get("content", "")
                        print(f"Response: {content[:100]}...")
                else:
                    print(f"❌ No choices in response for {model}")
            else:
                error_text = response.text
                if "does not support tool" in error_text.lower() or "tool" in error_text.lower():
                    print(f"❌ {model} does not support tools: {response.status_code}")
                else:
                    print(f"❌ Error with {model}: {response.status_code} - {error_text[:200]}")

        except Exception as e:
            print(f"❌ Exception testing {model}: {e}")

    print("\n❌ No HuggingFace models found with tool support")
    return False

def check_huggingface_models():
    """Check available models on HuggingFace (limited by API)"""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY not found")
        return

    print("\n📋 Checking HuggingFace models...")

    try:
        # HuggingFace doesn't have a simple models list endpoint like OpenRouter
        # We can check some known models manually
        known_models = [
            "microsoft/WizardLM-2-8x22B",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-2-7b-chat-hf",
            "microsoft/DialoGPT-medium"
        ]

        print("🔍 Checking known models for tool capabilities:")
        for model in known_models:
            url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}

            try:
                # Try to get model info
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    print(f"✅ {model} - Available")
                else:
                    print(f"❌ {model} - Not available or error")
            except:
                print(f"❌ {model} - Connection error")

    except Exception as e:
        print(f"❌ Error checking models: {e}")

if __name__ == "__main__":
    print("🚀 Testing HuggingFace API and Tool Support\n")

    if test_huggingface_api_key():
        check_huggingface_models()
        test_huggingface_chat_api()
    else:
        print("❌ Cannot proceed without valid API key")