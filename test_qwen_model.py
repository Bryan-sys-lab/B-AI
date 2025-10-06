#!/usr/bin/env python3
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Test the selected model
api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
if not api_key:
    print("No API key found")
    exit(1)

endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "qwen/qwen3-coder-480b-a35b-instruct",
    "messages": [{"role": "user", "content": "make a python function to represent bayes naive function"}],
    "temperature": 0.7,
}

print(f"Testing model: qwen/qwen3-coder-480b-a35b-instruct")
try:
    response = requests.post(endpoint, headers=headers, json=payload, timeout=30.0)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error response: {response.text}")
    else:
        data = response.json()
        print(f"Success! Content length: {len(data['choices'][0]['message']['content'])}")
        print(f"Content preview: {data['choices'][0]['message']['content'][:200]}...")
except Exception as e:
    print(f"Error: {e}")