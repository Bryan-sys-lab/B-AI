FROM python:3.9-slim
WORKDIR /app

RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    docker.io \
    build-essential \
    gcc \
    autoconf \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev \
    ca-certificates \
    ; \
    rm -rf /var/lib/apt/lists/*;

COPY sandbox_executor/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/
EXPOSE 8000

CMD uvicorn executor:app --host 0.0.0.0 --port $PORT