"""Examples for using the provider adapters.

This module contains simple illustrative snippets and is not executed as part
of the application. Keep examples minimal and avoid side effects.
"""

from .mistral_adapter import MistralAdapter
from .deepseek_adapter import DeepSeekAdapter
from .nim_adapter import NIMAdapter


def example_mistral():
    messages = [{"role": "user", "content": "Summarize the SOLID principles."}]
    adapter = MistralAdapter()
    resp = adapter.call_model(messages)
    print("Mistral response:", resp.text)


def example_deepseek():
    messages = [{"role": "user", "content": "Explain quantum computing briefly."}]
    adapter = DeepSeekAdapter()
    resp = adapter.call_model(messages)
    print("DeepSeek response:", resp.text)


def example_nim_thinkers():
    messages = [{"role": "user", "content": "Solve this complex problem."}]
    adapter = NIMAdapter(role="thinkers")
    resp = adapter.call_model(messages)
    print("NIM thinkers response:", resp.text)


if __name__ == "__main__":
    # Examples require API keys to be set in environment variables.
    # Uncomment to run locally for testing (not used in CI).
    # example_mistral()
    # example_deepseek()
    # example_nim_thinkers()
    pass