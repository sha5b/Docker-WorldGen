# syntax=docker/dockerfile:1

# We rely on PyTorch CUDA wheels (they bundle CUDA libs), so a slim Python base is fine.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_PORT=8000

# System deps for building and Open3D/graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone WorldGen
RUN git clone --depth 1 https://github.com/ZiYang-xie/WorldGen.git /opt/WorldGen
WORKDIR /opt/WorldGen
RUN git submodule update --init --recursive || true

# Install Torch/Torchvision with CUDA-enabled wheels
RUN python -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7.0" "torchvision>=0.22.0"

# Install PyTorch3D from community wheel index compatible with Torch 2.7/CUDA 12.8
# Ref: https://github.com/facebookresearch/pytorch3d/discussions/1752
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    pytorch3d

# Install WorldGen and its dependencies
# Note: WorldGen depends on some git packages and specific wheels.
RUN pip install .

# API server deps (include hf_transfer for fast HF downloads)
RUN pip install fastapi uvicorn[standard] python-multipart Pillow hf_transfer

# Add our API server
WORKDIR /app
COPY app/server.py /app/server.py

# Data directory for outputs
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# Health check to ensure server becomes ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${UVICORN_PORT}/healthz || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
