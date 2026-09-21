# Arcus — CUDA image for the RTX 5080 (Blackwell / sm_120). Mirrors boenet's docker setup.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    ARCUS_ENCODING=cl100k_base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# GPU PyTorch FIRST (cu128 wheels — Blackwell/sm_120 ready), so `pip install -e .`
# finds torch already satisfied and never pulls a CPU build.
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Install the package (pulls tiktoken, zstandard, numpy, pytest).
COPY pyproject.toml README.md ./
COPY arcus ./arcus
COPY evaluation ./evaluation
RUN pip3 install --no-cache-dir -e .

COPY scripts ./scripts
COPY tests ./tests

# Default: run the test suite. Override with a train command at `docker run`.
CMD ["python3", "-m", "pytest", "-q"]
