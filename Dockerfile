# Reproducible environment for the graf pipeline.
#
# Purpose: give a paper reviewer (or a future maintainer) a pinned
# environment where the fixture pipeline runs end-to-end without
# installing Python, PyTorch, or PyG on the host machine. This is
# not a deployment image; it is a reproducibility artifact.
#
# Build:  docker build -t graf:latest .
# Run:    docker run --rm graf:latest
#
# The default command runs the same test selection CI uses: the
# full unit suite minus the Hypothesis property tests, plus the
# 65% coverage floor.

FROM python:3.11-slim

WORKDIR /app

# Runtime libraries for OpenCV (opencv-python needs libGL and the
# GLib/X stack even in headless mode) and ffmpeg for any video I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from the committed requirement files, so
# the image stays in lockstep with CI and local dev environments.
COPY requirements requirements
RUN pip install --no-cache-dir -r requirements/base.txt \
    && pip install --no-cache-dir -r requirements/dev.txt \
    && pip install --no-cache-dir torch torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-geometric

# Copy the source tree. .dockerignore filters out .git, caches,
# local outputs, and non-fixture data.
COPY . .

ENV PYTHONPATH=/app/src

# Default: the same test selection CI runs. Override with
# `docker run --rm graf:latest python scripts/compare_baselines.py ...`
# to run a specific pipeline command inside the image.
CMD ["python", "-m", "pytest", "-q", \
     "--cov=src", "--cov=scripts", "--cov-report=term-missing", \
     "--cov-fail-under=65", "--ignore=tests/test_property_based.py"]
