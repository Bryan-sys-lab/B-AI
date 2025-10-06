from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

# Import our modules with fallbacks for local development
try:
    from metrics import get_metrics_response, record_request, SERVICE_UPTIME
except ImportError:
    # Fallback metrics implementation
    SERVICE_UPTIME = None
    def get_metrics_response():
        return "# Metrics not available - prometheus-client not installed"
    def record_request(method, endpoint, status, duration):
        pass

try:
    from tracing import setup_tracing, instrument_fastapi, get_tracer
except ImportError:
    # Fallback tracing implementation
    def setup_tracing():
        pass
    def instrument_fastapi(app):
        pass
    def get_tracer(name):
        return None

try:
    from health import get_health_status, get_readiness_status, get_liveness_status
except ImportError:
    # Fallback health implementation
    def get_health_status():
        return {"status": "healthy", "message": "Basic health check"}
    def get_readiness_status():
        return {"status": "ready", "message": "Basic readiness check"}
    def get_liveness_status():
        return {"status": "alive", "message": "Basic liveness check"}

try:
    from errors import get_recent_errors, capture_message
except ImportError:
    # Fallback error implementation
    def get_recent_errors(limit):
        return []
    def capture_message(message):
        pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tracing
setup_tracing()

# Create FastAPI app
app = FastAPI(title="Observability Service", description="Metrics, tracing, and monitoring service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with OpenTelemetry
instrument_fastapi(app)

# Get tracer
tracer = get_tracer(__name__)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    record_request(request.method, request.url.path, response.status_code, process_time)

    return response

@app.get("/")
async def root():
    return {"message": "Observability Service"}

@app.get("/health")
async def health_check():
    return get_health_status()

@app.get("/readiness")
async def readiness_check():
    return get_readiness_status()

@app.get("/liveness")
async def liveness_check():
    return get_liveness_status()

@app.get("/metrics")
async def metrics_endpoint():
    return Response(get_metrics_response(), media_type="text/plain; version=0.0.4; charset=utf-8")

@app.get("/test-tracing")
async def test_tracing():
    if tracer:
        with tracer.start_as_current_span("test-span"):
            logger.info("Test tracing endpoint called")
            time.sleep(0.1)  # Simulate some work
            return {"message": "Tracing test completed"}
    else:
        logger.info("Test tracing endpoint called (tracing not available)")
        time.sleep(0.1)  # Simulate some work
        return {"message": "Tracing test completed (tracing not available)"}

@app.get("/errors")
async def get_errors(limit: int = 50):
    return {"errors": get_recent_errors(limit)}

@app.get("/about")
async def about():
    return {
        "service": "observability",
        "version": "1.0.0",
        "description": "Observability service with metrics, tracing, and monitoring"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)