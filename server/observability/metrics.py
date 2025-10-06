from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

SERVICE_UPTIME = Gauge(
    'service_uptime_seconds',
    'Service uptime in seconds'
)

ERROR_COUNT = Counter(
    'errors_total',
    'Total number of errors',
    ['type', 'endpoint']
)

# Custom business metrics
TASKS_PROCESSED = Counter(
    'tasks_processed_total',
    'Total number of tasks processed',
    ['type']
)

TASK_PROCESSING_TIME = Histogram(
    'task_processing_duration_seconds',
    'Task processing time in seconds',
    ['type']
)

def get_metrics_response():
    """Return Prometheus metrics as HTTP response"""
    return generate_latest()

def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

def record_error(error_type: str, endpoint: str = ""):
    """Record error metrics"""
    ERROR_COUNT.labels(type=error_type, endpoint=endpoint).inc()

def record_task_processed(task_type: str, duration: float = None):
    """Record task processing metrics"""
    TASKS_PROCESSED.labels(type=task_type).inc()
    if duration is not None:
        TASK_PROCESSING_TIME.labels(type=task_type).observe(duration)

def update_uptime(start_time: float):
    """Update service uptime metric"""
    SERVICE_UPTIME.set(time.time() - start_time)