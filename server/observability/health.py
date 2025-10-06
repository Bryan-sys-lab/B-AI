import time
import psutil
import os
from typing import Dict, Any
from metrics import SERVICE_UPTIME, record_error

START_TIME = time.time()

def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    try:
        # Basic health info
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - START_TIME,
            "service": "observability"
        }

        # System metrics
        health_data["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }

        # Update uptime metric
        SERVICE_UPTIME.set(health_data["uptime_seconds"])

        return health_data

    except Exception as e:
        record_error("health_check_failed", "health")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

def get_readiness_status() -> Dict[str, Any]:
    """Check if service is ready to serve requests"""
    # For now, just check if we can access basic resources
    try:
        # Check if we can write to a temp file (basic I/O check)
        with open('/tmp/health_check', 'w') as f:
            f.write('ok')
        os.remove('/tmp/health_check')

        return {
            "status": "ready",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e),
            "timestamp": time.time()
        }

def get_liveness_status() -> Dict[str, Any]:
    """Check if service is alive (basic liveness probe)"""
    return {
        "status": "alive",
        "timestamp": time.time(),
        "uptime_seconds": time.time() - START_TIME
    }