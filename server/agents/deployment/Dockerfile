FROM python:3.11-slim

WORKDIR /app

# Install minimal OS build dependencies
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    autoconf \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev \
    ca-certificates \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    true

COPY agents/deployment/requirements.txt /tmp/requirements.txt
RUN for i in 1 2 3; do \
    pip --default-timeout=120 install --no-cache-dir -r /tmp/requirements.txt && break || true; \
    echo "pip install failed, attempt $i/3"; \
    sleep $((i * 3)); \
    done

# Copy the entire repository into the image so the service can import sibling
# packages (providers, tool_api_gateway, orchestrator, ...). This mirrors the
# orchestrator Dockerfile which copies the repo at /app.
COPY . /app/

EXPOSE 8000

# Ensure the repo root is on PYTHONPATH so package imports like
# `agents.deployment.*` resolve correctly.
ENV PYTHONPATH=/app

# Run the application using the package-qualified module path so relative
# imports inside the package (e.g., `from .prompt_builder import ...`) work.
CMD ["uvicorn", "agents.deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]