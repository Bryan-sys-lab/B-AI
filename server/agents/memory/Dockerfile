FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent code
COPY main.py .

# Expose port
EXPOSE 8025

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8025/health || exit 1

# Run the agent
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8025"]