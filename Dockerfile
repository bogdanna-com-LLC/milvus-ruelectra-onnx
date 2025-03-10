# BUILDER STAGE
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables to reduce image size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and other Python dependencies
COPY app/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Cache the model in the builder stage
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; \
    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/ruElectra-small'); \
    model = AutoModel.from_pretrained('sberbank-ai/ruElectra-small')"

# RUNTIME STAGE
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cached models from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY app /app

# Clean up unnecessary files
RUN find /usr/local -name '*.pyc' -delete \
    && find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true \
    && find /root/.cache/huggingface -name "*.h5" -delete \
    && find /root/.cache/huggingface -name "*.msgpack" -delete \
    && find /root/.cache/huggingface -name "*.ot" -delete

# Expose port for FastAPI
EXPOSE 8000

# Start the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]