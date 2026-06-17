# CPU-only image for carotid / venous training (no NVIDIA CUDA required)
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY fedsm/ ./fedsm/
COPY scripts/ ./scripts/
COPY app.py run_all.py pyproject.toml requirements.txt ./

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

CMD ["python", "app.py", "carotid"]
