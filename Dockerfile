# ══════════════════════════════════════════════════════════════
# Tirumala AI Assistant — Production Dockerfile
#
# Build:   docker build -t tirumala-ai:latest .
# Run:     docker run -p 8000:8000 --env-file .env tirumala-ai:latest
#
# GPU note: To enable GPU passthrough, swap the base image to
#           nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
#           and install torch with the CUDA index URL instead.
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# Stage 1 — builder: install all Python deps
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip once in the builder stage
RUN pip install --upgrade pip

# ── CPU-only PyTorch (swap URL for CUDA if needed) ──────────
# This is installed separately so Docker layer-caching skips it
# on re-builds when only source code changes.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── Project dependencies ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2 — runtime: lean production image
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="Mahesh1827"
LABEL description="Tirumala Multi-Agent AI Assistant — FastAPI service"
LABEL version="1.0.0"

# System runtime libraries required by PaddleOCR and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        poppler-utils \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code (excludes paths in .dockerignore)
COPY --chown=appuser:appuser . .

# Create cache directories with correct permissions
RUN mkdir -p /app/.cache/model_cache && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# ── Environment defaults (secrets must come from --env-file or -e) ──
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/model_cache/huggingface \
    HF_HUB_CACHE=/app/.cache/model_cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/model_cache/huggingface/hub \
    TORCH_HOME=/app/.cache/model_cache/torch \
    PPOCR_HOME=/app/.cache/model_cache/paddleocr \
    PADDLE_HOME=/app/.cache/model_cache/paddle \
    PORT=8000

EXPOSE 8000

# ── Health check (Docker daemon will mark container unhealthy if it fails) ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ── Entry point ──────────────────────────────────────────────
# --workers 1: safe default (increase for multi-core, but sentence-transformers
#              is not fork-safe with multiple workers unless you pre-load models)
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
