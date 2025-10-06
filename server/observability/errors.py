import logging
import json
import time
from typing import Dict, Any, Optional
from metrics import record_error

logger = logging.getLogger(__name__)

class ErrorTracker:
    def __init__(self):
        self.errors = []
        self.max_errors = 1000  # Keep last 1000 errors

    def capture_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, str]] = None):
        """Capture an exception with context"""
        error_data = {
            "timestamp": time.time(),
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "context": context or {},
            "tags": tags or {},
            "stack_trace": self._get_stack_trace(exception)
        }

        self.errors.append(error_data)

        # Keep only recent errors
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)

        # Record in metrics
        record_error("exception", tags.get("endpoint", "") if tags else "")

        # Log the error
        logger.error(f"Exception captured: {error_data['exception_type']}: {error_data['message']}", extra=error_data)

        return error_data

    def capture_message(self, message: str, level: str = "error", context: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, str]] = None):
        """Capture a custom error message"""
        error_data = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "context": context or {},
            "tags": tags or {},
            "type": "message"
        }

        self.errors.append(error_data)

        if len(self.errors) > self.max_errors:
            self.errors.pop(0)

        # Record in metrics
        record_error(level, tags.get("endpoint", "") if tags else "")

        # Log the message
        log_func = getattr(logger, level, logger.error)
        log_func(f"Message captured: {message}", extra=error_data)

        return error_data

    def get_recent_errors(self, limit: int = 50) -> list:
        """Get recent errors"""
        return self.errors[-limit:]

    def get_errors_by_type(self, error_type: str) -> list:
        """Get errors by type"""
        return [e for e in self.errors if e.get("exception_type") == error_type or e.get("level") == error_type]

    def _get_stack_trace(self, exception: Exception) -> str:
        """Get formatted stack trace"""
        import traceback
        return "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))

# Global error tracker instance
error_tracker = ErrorTracker()

def capture_exception(exception: Exception, context: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, str]] = None):
    """Convenience function to capture exceptions"""
    return error_tracker.capture_exception(exception, context, tags)

def capture_message(message: str, level: str = "error", context: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, str]] = None):
    """Convenience function to capture messages"""
    return error_tracker.capture_message(message, level, context, tags)

def get_recent_errors(limit: int = 50) -> list:
    """Get recent errors"""
    return error_tracker.get_recent_errors(limit)