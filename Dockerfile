# First stage: Convert PyTorch model to ONNX
FROM python:3.10-slim AS converter

LABEL maintainer="Danila developer.permogorsky@gmail.com"
LABEL version="1.0"
LABEL description="ruElectra embedding model with ONNX Runtime (CPU-optimized)"

WORKDIR /convert

# Install required packages for conversion
# Split installation to use PyTorch index only for torch
RUN pip install --no-cache-dir onnx==1.14.0 transformers==4.30.2 protobuf==3.20.3
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Copy conversion script
COPY convert_to_onnx.py .

# Run conversion
RUN python convert_to_onnx.py

# Second stage: Build the runtime image with ONNX (CPU version)
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create model and data directories
RUN mkdir -p /app/models /app/data

# Install only the required packages for inference
# Added pymilvus to the packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir fastapi==0.103.1 uvicorn==0.23.2 pydantic==2.3.0 && \
    pip install --no-cache-dir onnxruntime==1.15.1 && \
    pip install --no-cache-dir transformers==4.30.2 && \
    pip install --no-cache-dir pymilvus && \
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