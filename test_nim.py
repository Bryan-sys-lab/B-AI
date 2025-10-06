#!/usr/bin/env python3
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Test NIM API call
api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
if not api_key:
    print("No API key found")
    exit(1)

print(f"API key found: {api_key[:4]}...{api_key[-4:]}")

endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "make a python function to represent bayes naive function"}],
    "temperature": 0.7,
}

print(f"Making request to {endpoint}")
try:
    response = requests.post(endpoint, headers=headers, json=payload, timeout=30.0)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Response keys: {list(data.keys())}")
        if "choices" in data and data["choices"]:
            print(f"Content: {data['choices'][0]['message']['content'][:200]}...")
except Exception as e:
    print(f"Error: {e}")