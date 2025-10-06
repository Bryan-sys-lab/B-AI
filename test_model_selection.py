#!/usr/bin/env python3
from server.providers.model_registry import choose_model_for_role

model = choose_model_for_role("builders")
print(f"Selected model for 'builders' role: {model}")