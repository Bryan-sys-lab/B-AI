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

COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel; \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/

# Ensure the repo root is on PYTHONPATH so imports like 'orchestrator.xxx' work
ENV PYTHONPATH=/app

EXPOSE 8000

# Run the app via package-qualified module name
CMD uvicorn orchestrator.main:app --host 0.0.0.0 --port $PORT