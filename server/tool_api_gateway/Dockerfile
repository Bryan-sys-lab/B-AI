FROM python:3.9-slim

WORKDIR /app

# Install minimal build deps first to allow building wheels for packages that require compilation.
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

COPY tool_api_gateway/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the whole repository so the package `tool_api_gateway` can import
# sibling packages (providers, sandbox_executor, etc.) when run inside the image.
COPY . /app/

EXPOSE 8000


ENV PYTHONPATH=/app
CMD uvicorn tool_api_gateway.main:app --host 0.0.0.0 --port $PORT