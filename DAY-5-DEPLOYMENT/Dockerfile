# =============================================================================
# Day 5 — Production Dockerfile (Multi-stage)
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Day-5-MLOps"
LABEL description="ML Model Serving API"

WORKDIR /app

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages
COPY --from=builder /install /usr/local

# Copy application code
COPY deployment/ ./deployment/
COPY monitoring/ ./monitoring/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY .env.example .env.example

# Create directories with proper permissions (as root)
RUN mkdir -p logs models/v1 && \
    chmod -R 777 logs && \
    chmod -R 755 models

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command (single worker to avoid permission issues)
CMD ["uvicorn", "deployment.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]