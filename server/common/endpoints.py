"""Common FastAPI endpoint implementations."""

from fastapi import FastAPI
from providers.system_prompt import CANNED_RESPONSES


def add_health_endpoint(app: FastAPI):
    """Add standard health endpoint to FastAPI app."""
    @app.get("/health")
    def health():
        return {"status": "ok"}


def add_about_endpoint(app: FastAPI):
    """Add standard about endpoint to FastAPI app."""
    @app.get("/about")
    def about(detail: str = "short"):
        level = (detail or "").lower()
        if level not in ("short", "medium", "detailed"):
            return {"error": "detail must be one of: short, medium, detailed"}
        resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
        return {"level": level, "response": resp}