import requests
import time
from comparator_service.config import SANDBOX_EXECUTOR_URL
from tool_api_gateway.models import ShellExecRequest

async def run_performance_check(candidate, repo_url, branch):
    """
    Run performance checks, e.g., benchmark execution time.
    """
    # Assume a benchmark command, e.g., run tests with timing
    benchmark_cmd = "time python -m pytest --tb=no -q /workspace"  # Example
    request = ShellExecRequest(command="sh", args=["-c", benchmark_cmd], working_dir="/workspace", timeout=60)
    start_time = time.time()
    # We don't need the response object here; measure wall-clock time for a simple benchmark.
    requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=request.model_dump())
    end_time = time.time()
    execution_time = end_time - start_time
    # Assume baseline time is 10s, impact = (time - 10)/10, clamped
    baseline = 10.0
    impact = max(0, (execution_time - baseline) / baseline)
    performance_impact = min(impact, 1.0)  # 0-1

    return {"performance_impact": performance_impact, "execution_time": execution_time}