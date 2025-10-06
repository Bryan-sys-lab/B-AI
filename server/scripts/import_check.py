import sys
import importlib

# Ensure workspace root is on path
sys.path.insert(0, "/home/su/Aetherium/B1.0")

modules = ["comparator_service.main", "orchestrator.main"]
results = {}
for m in modules:
    try:
        importlib.import_module(m)
        results[m] = 'imported'
    except Exception as e:
        results[m] = repr(e)

print(results)
