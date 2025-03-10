
# First stage: Convert PyTorch model to ONNX
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS converter

LABEL maintainer="Danila developer.permogorsky@gmail.com"
LABEL version="1.0"
LABEL description="ruElectra embedding model with ONNX Runtime"

WORKDIR /convert

# Install required packages for conversion
# Explicitly install onnx first to avoid the dependency error
RUN pip install --no-cache-dir onnx==1.14.0 transformers==4.30.2 protobuf==3.20.3

# Copy conversion script
COPY convert_to_onnx.py .

# Run conversion
RUN python convert_to_onnx.py

# Second stage: Build the runtime image with ONNX
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python and required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create model directory
RUN mkdir -p /app/models

# Install only the required packages for inference
# Install only the required packages for inference with BuildKit cache mounting
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install numpy==1.24.3 && \
    pip install fastapi==0.103.1 uvicorn==0.23.2 pydantic==2.3.0 && \
    pip install onnxruntime-gpu==1.15.1 && \
    pip install transformers==4.30.2 && \
    find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Copy ONNX model and tokenizer from converter stage
COPY --from=converter /convert/ruelectra_small.onnx /app/models/
COPY --from=converter /convert/onnx_tokenizer /app/models/onnx_tokenizer

# Copy application code
COPY app /app

# Cleanup
RUN find /usr/local -name '*.pyc' -delete \
    && find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true \
    && pip cache purge 

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]