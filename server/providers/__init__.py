"""Provider adapters for CodeAgent system.

This package exposes adapters. Avoid importing submodules at top-level to
prevent circular imports when the package is imported from different
contexts (e.g., service containers). Consumers should import the specific
adapter they need, e.g. `from providers.nim_adapter import NIMAdapter`.
"""

from .base_adapter import BaseAdapter, ModelResponse

__all__ = ["BaseAdapter", "ModelResponse"]