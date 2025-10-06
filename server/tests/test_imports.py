import importlib

MODULES = [
    "orchestrator.main",
    "agents.fix_implementation.main",
    "providers.nim_adapter",
    "comparator_service.main",
    "tool_api_gateway.main",
]


def test_imports():
    failed = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append((m, type(e).__name__, str(e)))
    assert not failed, f"Import failures: {failed}\n"
