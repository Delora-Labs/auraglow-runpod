# ============================================================================
# RunPod Serverless Worker for SDXL Image Generation
# Base: PyTorch with CUDA 12.1
# ============================================================================

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV DIFFUSERS_CACHE=/runpod-volume/huggingface

# Set working directory
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Pre-download the model during build (optional but recommended)
# This bakes the model into the image, increasing size but reducing cold start
ARG DOWNLOAD_MODEL=true
ARG MODEL_ID=SG161222/RealVisXL_V5.0

RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python -c "from diffusers import StableDiffusionXLPipeline; \
    StableDiffusionXLPipeline.from_pretrained('${MODEL_ID}', \
    torch_dtype='float16', variant='fp16', use_safetensors=True)"; \
    fi

# Copy handler code
COPY handler.py /handler.py

# Create cache directories
RUN mkdir -p /tmp/lora_cache

# Set the entrypoint
CMD ["python", "-u", "/handler.py"]
