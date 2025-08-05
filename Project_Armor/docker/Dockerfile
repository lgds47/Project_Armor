# Dockerfile for J&J Contact Lens Defect Detection
# Optimized for Nebius Cloud GPU instances (A100)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libnotify-dev \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY armor_pipeline/ /app/armor_pipeline/
COPY config/ /app/config/
COPY model_zoo/ /app/model_zoo/
COPY scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/data \
    /app/output/checkpoints \
    /app/output/evaluation_results \
    /app/output/visualizations \
    /app/logs

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "Starting J&J Contact Lens Defect Detection Pipeline"\n\
echo "CUDA Available: $(python3.11 -c "import torch; print(torch.cuda.is_available())")"\n\
echo "GPU Count: $(python3.11 -c "import torch; print(torch.cuda.device_count())")"\n\
exec python3.11 -m armor_pipeline.cli "$@"' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (show help)
CMD ["--help"]