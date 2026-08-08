# ═══════════════════════════════════════════════════════════════════════════════
# StockBuddy — Production Dockerfile (Multi-Stage Build)
# ═══════════════════════════════════════════════════════════════════════════════

# --- Stage 1: Build Dependencies ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime Environment ---
FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    LOG_FORMAT=json

# Install runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Security hardening: Create non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

# Copy application source code
COPY --chown=appuser:appgroup . /app

# Ensure model artifacts and cache directories exist with proper permissions
RUN mkdir -p /app/model_artifacts /app/inference_cache && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 5000

# Container Healthcheck (Liveness Probe)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server startup with Gunicorn (4 workers)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "app:flask_app"]
