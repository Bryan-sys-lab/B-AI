FROM python:3.9-slim

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

COPY comparator_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "comparator_service.main:app", "--host", "0.0.0.0", "--port", "8000"]