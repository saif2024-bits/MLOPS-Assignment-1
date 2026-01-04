"""
Monitoring and metrics collection for Heart Disease Prediction API
Provides Prometheus metrics and request tracking
"""

import time
from typing import Callable
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
import logging

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    registry=registry,
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['prediction_class'],
    registry=registry
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction confidence scores',
    registry=registry,
    buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0)
)

BATCH_SIZE = Histogram(
    'batch_prediction_size',
    'Size of batch predictions',
    registry=registry,
    buckets=(1, 5, 10, 20, 50, 100)
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model',
    registry=registry
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of requests currently being processed',
    registry=registry
)

ERROR_COUNT = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['error_type', 'endpoint'],
    registry=registry
)

HEALTH_STATUS = Gauge(
    'api_health_status',
    'API health status (1 = healthy, 0 = unhealthy)',
    registry=registry
)


class PrometheusMiddleware:
    """Middleware to collect Prometheus metrics for all requests"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request information
        method = scope["method"]
        path = scope["path"]

        # Skip metrics endpoint to avoid recursion
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        # Increment active requests
        ACTIVE_REQUESTS.inc()

        # Track request duration
        start_time = time.time()
        status_code = 500  # Default to error

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            ERROR_COUNT.labels(error_type=type(e).__name__, endpoint=path).inc()
            logger.error(f"Error processing request: {e}")
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
            ACTIVE_REQUESTS.dec()

            # Log request
            logger.info(
                f"{method} {path} - Status: {status_code} - Duration: {duration:.3f}s"
            )


def track_prediction(prediction: int, confidence: float = None):
    """Track a prediction in Prometheus metrics"""
    prediction_class = "heart_disease" if prediction == 1 else "no_disease"
    PREDICTION_COUNT.labels(prediction_class=prediction_class).inc()

    if confidence is not None:
        PREDICTION_CONFIDENCE.observe(confidence)


def track_batch_prediction(batch_size: int):
    """Track batch prediction size"""
    BATCH_SIZE.observe(batch_size)


def set_model_load_time(duration: float):
    """Set the model load time metric"""
    MODEL_LOAD_TIME.set(duration)


def set_health_status(is_healthy: bool):
    """Set the health status metric"""
    HEALTH_STATUS.set(1 if is_healthy else 0)


def metrics_endpoint():
    """Generate Prometheus metrics in the expected format"""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )
