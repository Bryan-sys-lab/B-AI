from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import logging

logger = logging.getLogger(__name__)

def setup_tracing(service_name: str = "observability-service", jaeger_host: str = "localhost", jaeger_port: int = 6831):
    """Initialize OpenTelemetry tracing"""
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider())

    # Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )
    jaeger_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(jaeger_processor)

    # Console exporter for debugging
    console_exporter = ConsoleSpanExporter()
    console_processor = BatchSpanProcessor(console_exporter)
    trace.get_tracer_provider().add_span_processor(console_processor)

    logger.info(f"Tracing initialized for service: {service_name}")

def instrument_fastapi(app):
    """Instrument FastAPI application with OpenTelemetry"""
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumented with OpenTelemetry")

def get_tracer(name: str):
    """Get a tracer instance"""
    return trace.get_tracer(name)