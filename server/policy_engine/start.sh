#!/bin/bash

# Start OPA server in the background
opa run --server --addr :8181 --log-level info &

# Wait for OPA to be ready
sleep 5

# Start the FastAPI app
uvicorn main:app --host 0.0.0.0 --port 8000