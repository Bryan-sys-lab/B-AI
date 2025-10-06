#!/usr/bin/env python3
"""Test script for prompt cache functionality."""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from providers.prompt_cache import prompt_cache_manager

def test_cache():
    print("Testing prompt cache...")

    # Test messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    # Test storing
    print("Storing response...")
    cache_key = prompt_cache_manager.store_response(
        provider="test",
        model="test-model",
        role="default",
        messages=messages,
        response={"text": "I'm doing well, thank you!"},
        tokens_used=10,
        latency_ms=100,
        cost_estimate=0.001
    )
    print(f"Stored with cache key: {cache_key}")

    # Test retrieving
    print("Retrieving response...")
    cached = prompt_cache_manager.get_cached_response(
        provider="test",
        model="test-model",
        role="default",
        messages=messages
    )

    if cached:
        print(f"Cache hit! Response: {cached['response']['text']}")
    else:
        print("Cache miss!")

    # Test stats
    print("Getting cache stats...")
    stats = prompt_cache_manager.get_cache_stats()
    print(f"Cache stats: {stats}")

if __name__ == "__main__":
    test_cache()