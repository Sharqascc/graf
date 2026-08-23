FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements requirements

RUN pip install --no-cache-dir -r requirements/base.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-geometric pytest pytest-cov

COPY . .

ENV PYTHONPATH=/app/src

CMD ["pytest", "-q"]
