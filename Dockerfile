# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Base image
# Pinned to the Debian codename (bookworm) on purpose: the "python:3.11-slim"
# tag floats to whatever Debian release is newest (currently trixie), so an
# unpinned tag can change the OS underneath you between builds. Pinning keeps
# a rebuild reproducible weeks later. amd64 and arm64 are both published, so
# this works on Apple Silicon laptops and x86 EC2 instances alike.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# PYTHONDONTWRITEBYTECODE: don't litter the image with .pyc files.
# PYTHONUNBUFFERED:        logs appear immediately in `docker logs`.
# PIP_NO_CACHE_DIR:        don't leave the pip download cache in the image.
# MODEL_PATH:              explicit location of the trained artifact.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/model/customer_model.pkl

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies first, code second.
# This layer is only rebuilt when requirements.txt changes, so everyday code
# edits reuse the cached (slow) dependency install instead of reinstalling
# scikit-learn and scipy on every build.
# ---------------------------------------------------------------------------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Application code and the trained model artifact.
# train_model.py is deliberately NOT copied: it is a build/development tool,
# not something the serving container needs. Re-run it on your machine to
# regenerate model/customer_model.pkl, then rebuild this image.
# ---------------------------------------------------------------------------
COPY app/ ./app/
COPY model/ ./model/

# ---------------------------------------------------------------------------
# Run as a non-root user. If the process is ever compromised it cannot write
# to the application code or the model artifact.
# ---------------------------------------------------------------------------
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# Build-time gate. If model/customer_model.pkl is missing, unreadable or
# malformed, the BUILD fails here with a clear message. Without this check the
# image would build happily and then crash-loop on EC2 at 3am.
# ---------------------------------------------------------------------------
RUN python -c "from app.model import SegmentationModel; model = SegmentationModel.load(); print(f'Model smoke test OK: {model.n_clusters} clusters, {model.n_features} features')"

EXPOSE 8000

# ---------------------------------------------------------------------------
# Container health probe. curl is not installed in slim images, so this uses
# the standard library. It fails (non-zero) if the service is unreachable or
# if the model is not loaded, and `docker inspect` reports "unhealthy".
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import json, urllib.request; d = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)); raise SystemExit(0 if d['status'] == 'ok' and d['model_loaded'] else 1)"

# ---------------------------------------------------------------------------
# Bind to 0.0.0.0, not 127.0.0.1: inside a container, localhost is only
# reachable from inside that container, so a port mapping would appear dead.
# ---------------------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]