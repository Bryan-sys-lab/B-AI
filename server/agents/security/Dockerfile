FROM python:3.11-slim

WORKDIR /app

# Install minimal OS build dependencies and security tools
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
    git \
    openjdk-11-jre-headless \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    true

# Install Semgrep
RUN pip install semgrep

# Install Trivy
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - && \
    echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | tee -a /etc/apt/sources.list.d/trivy.list && \
    apt-get update && \
    apt-get install -y trivy && \
    rm -rf /var/lib/apt/lists/*

# Install OWASP Dependency-Check
RUN wget -O /tmp/dependency-check.zip https://github.com/jeremylong/DependencyCheck/releases/download/v8.4.0/dependency-check-8.4.0-release.zip && \
    unzip /tmp/dependency-check.zip -d /opt && \
    ln -s /opt/dependency-check/bin/dependency-check.sh /usr/local/bin/dependency-check && \
    rm /tmp/dependency-check.zip

COPY agents/security/requirements.txt /tmp/requirements.txt
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
# `agents.security.*` resolve correctly.
ENV PYTHONPATH=/app

# Run the application using the package-qualified module path so relative
# imports inside the package (e.g., `from .vulnerability_scanner import ...`) work.
CMD uvicorn agents.security.main:app --host 0.0.0.0 --port $PORT