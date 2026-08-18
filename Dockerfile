# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# System dependencies:
# - build-essential: some pip packages (e.g. chromadb's dependencies) need
#   a C compiler if a prebuilt wheel isn't available for the target arch
# - curl: used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first, separately from app code, so Docker's
# layer cache avoids reinstalling everything on every code change — only
# rebuilds this layer when requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

# Never bake secrets into the image. Real values are supplied at
# `docker run` time via --env-file .env or -e, or via docker-compose's
# env_file directive (see docker-compose.yml). This Dockerfile only copies
# .env.example (the template) if present — actual .env should be excluded
# via .dockerignore, and is.

EXPOSE 8501

# Streamlit exposes a built-in health endpoint; used by both this
# HEALTHCHECK and docker-compose's healthcheck for orchestration platforms
# (Docker Swarm, Kubernetes liveness probes, etc.) that key off container
# health status.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# CHECKPOINTER_BACKEND=sqlite works correctly here (unlike Streamlit
# Community Cloud) as long as you mount a volume for the sqlite file —
# see docker-compose.yml's `volumes:` section — since this container's
# filesystem is NOT ephemeral in the same way a managed PaaS deployment is
# (it persists for the container's lifetime, and via a volume, beyond it).
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
