# CPU-only image for carotid / venous training (no NVIDIA CUDA required)
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU wheels (do not install CUDA builds)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Project code (full repo mounted at runtime via docker compose / -v)
COPY *.py ./

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Default: carotid train + validate (override CMD as needed)
CMD ["python", "run_carotid.py"]
