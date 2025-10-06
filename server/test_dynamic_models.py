#!/usr/bin/env python3
"""Test script for dynamic model loading."""

import sys
import os

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from providers.model_registry import get_available_models, choose_model_for_role

def test_get_available_models():
    """Test fetching available models."""
    print("Testing get_available_models...")
    models = get_available_models()
    print(f"Available models: {len(models)} found")
    if models:
        print(f"First 5: {models[:5]}")
    else:
        print("No models available (check API key)")
    return models

def test_choose_model_for_role():
    """Test model selection for roles."""
    print("\nTesting choose_model_for_role...")
    roles = ["default", "thinkers", "builders"]
    for role in roles:
        model = choose_model_for_role(role)
        print(f"Role '{role}': selected model '{model}'")

if __name__ == "__main__":
    models = test_get_available_models()
    test_choose_model_for_role()
    print("\nTest completed.")