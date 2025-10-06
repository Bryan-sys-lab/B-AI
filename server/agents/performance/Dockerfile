FROM python:3.11-slim

WORKDIR /app

# Install minimal OS build dependencies plus Py-Spy for profiling
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
    wget \
    curl \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    true

# Install Py-Spy
RUN wget https://github.com/benfred/py-spy/releases/download/v0.3.14/py-spy_0.3.14_amd64.deb && \
    dpkg -i py-spy_0.3.14_amd64.deb && \
    rm py-spy_0.3.14_amd64.deb

COPY agents/performance/requirements.txt /tmp/requirements.txt
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
# `agents.performance.*` resolve correctly.
ENV PYTHONPATH=/app

# Run the application using the package-qualified module path so relative
# imports inside the package (e.g., `from .profiler import ...`) work.
CMD ["uvicorn", "agents.performance.main:app", "--host", "0.0.0.0", "--port", "8000"]