import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible so adapters and
# other modules that read environment variables (e.g. providers/*) will see
# the values when they're imported or instantiated.
load_dotenv()

# Ensure repo root is on sys.path before any other imports so local modules
# (e.g., prompt_builder) and top-level packages resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    # Intentional import-time sys.path modification to ensure repo root is
    # available in containerized or subdirectory run contexts. Lint rule E402
    # (module level import not at top) is intentionally suppressed here.
    sys.path.insert(0, _repo_root)  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import List, Optional, Dict, Any  # noqa: E402
import logging  # noqa: E402
import psutil  # noqa: E402
import time  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES  # noqa: E402

app = FastAPI(title="Monitoring Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringRequest(BaseModel):
    target_service: str
    metrics_type: str = "system"  # system, application, performance
    time_range: str = "5m"  # 5m, 1h, 24h
    alert_thresholds: Optional[Dict[str, Any]] = None

class ExecuteRequest(BaseModel):
    description: str

class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, Any]
    network_io: Dict[str, Any]
    timestamp: float

class ApplicationMetrics(BaseModel):
    response_time: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[float] = None
    active_connections: Optional[int] = None

class MonitoringResponse(BaseModel):
    system_metrics: SystemMetrics
    application_metrics: Optional[ApplicationMetrics] = None
    alerts: List[str] = []
    recommendations: List[str] = []
    error: Optional[str] = None

def get_system_metrics() -> SystemMetrics:
    """Collect basic system metrics."""
    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(interval=1),
        memory_percent=psutil.virtual_memory().percent,
        disk_usage={
            "total": psutil.disk_usage('/').total,
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        },
        network_io={
            "bytes_sent": psutil.net_io_counters().bytes_sent,
            "bytes_recv": psutil.net_io_counters().bytes_recv,
            "packets_sent": psutil.net_io_counters().packets_sent,
            "packets_recv": psutil.net_io_counters().packets_recv
        },
        timestamp=time.time()
    )

@app.post("/monitor", response_model=MonitoringResponse)
async def monitor_service(request: MonitoringRequest):
    try:
        logger.info(f"Monitoring {request.target_service} for {request.metrics_type} metrics")

        # Get system metrics
        system_metrics = get_system_metrics()

        # Initialize application metrics
        application_metrics = None
        alerts = []
        recommendations = []

        # Check for system-level alerts
        if system_metrics.cpu_percent > 80:
            alerts.append(f"High CPU usage: {system_metrics.cpu_percent}%")
            recommendations.append("Consider scaling resources or optimizing CPU-intensive processes")

        if system_metrics.memory_percent > 85:
            alerts.append(f"High memory usage: {system_metrics.memory_percent}%")
            recommendations.append("Monitor memory leaks or increase memory allocation")

        if system_metrics.disk_usage["percent"] > 90:
            alerts.append(f"Low disk space: {system_metrics.disk_usage['percent']}% used")
            recommendations.append("Clean up disk space or add storage capacity")

        # For application-specific monitoring, we would typically query service endpoints
        # This is a simplified version
        if request.metrics_type == "application":
            application_metrics = ApplicationMetrics(
                response_time=None,  # Would need to query actual service
                error_rate=None,
                throughput=None,
                active_connections=None
            )

        return MonitoringResponse(
            system_metrics=system_metrics,
            application_metrics=application_metrics,
            alerts=alerts,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Error in monitoring: {str(e)}")
        return MonitoringResponse(
            system_metrics=get_system_metrics(),
            alerts=[],
            recommendations=[],
            error=str(e)
        )

@app.post("/execute")
async def execute_task(request: ExecuteRequest):
    try:
        logger.info(f"Executing monitoring task: {request.description}")
        desc = (request.description or "").strip()

        # Simple canned response still allowed for trivial 'hello'
        if desc.lower() == "hello":
            return {"result": "Hello, world!", "success": True}

        # Determine if this is a monitoring task
        is_monitoring_task = any(keyword in desc.lower() for keyword in ["monitor", "observability", "metrics", "performance", "health", "alert"])

        # Choose appropriate system prompt based on task type
        if is_monitoring_task:
            system_prompt = (
                "You are a monitoring and observability expert. Provide insights on system performance, "
                "application metrics, alerting strategies, and monitoring best practices. "
                "Focus on reliability, scalability, and proactive issue detection."
            )
        else:
            system_prompt = SYSTEM_PROMPT

        # Use NVIDIA NIM exclusively for model calls
        try:
            adapter = NIMAdapter(role="builders")
        except Exception as e:
            msg = f"NIM adapter initialization failed: {e}. Ensure NVIDIA_NIM_API_KEY is set in the environment."
            logger.error(msg)
            return {"error": msg}

        # Include appropriate system prompt for better responses
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": desc}]
        try:
            response = adapter.call_model(messages, temperature=0.7 if is_monitoring_task else 0.2)
        except Exception as e:
            logger.error("NIM provider call failed: %s", str(e))
            return {"error": f"NIM provider call failed: {e}"}

        # Return the provider's textual response and structured data if any
        return {
            "result": response.text,
            "success": True,
            "tokens": response.tokens,
            "latency_ms": response.latency_ms,
            "structured": response.structured_response or {},
            "is_monitoring_task": is_monitoring_task,
        }
    except Exception as e:
        logger.error(f"Error executing monitoring task: {str(e)}")
        return {"error": str(e)}


@app.get("/about")
def about(detail: Optional[str] = "short"):
    """Return a canned "about" response at three levels: short, medium, detailed.

    Also return the current `SYSTEM_PROMPT` so operators can inspect how the
    agent is being primed.
    """
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {
        "level": level,
        "response": resp,
        "response": resp,
    }