# ==============================================================================
# VYREX AI - Production Container Image
# Multi-stage build with NVIDIA CUDA 12 support, PyTorch, and FastAPI / vLLM
# Architect: Suborno Roy Chowdhury | VYREX AI Project
# ==============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install production API & inference dependencies
COPY pyproject.toml /app/
RUN pip install \
    transformers>=4.48.0 \
    peft>=0.14.0 \
    accelerate>=1.2.0 \
    bitsandbytes>=0.45.0 \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.32.0 \
    pydantic>=2.10.0 \
    sse-starlette>=2.2.0 \
    httpx>=0.28.0

# Copy source code and scripts
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY pretrained/ /app/pretrained/

# Install local package in editable mode
RUN pip install -e /app

# Expose standard REST API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV VYREX_BACKEND=hf

# Start the OpenAI-compatible REST API server
CMD ["python", "scripts/serve_vyrex_api.py", "--host", "0.0.0.0", "--port", "8000"]
