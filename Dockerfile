# ---------------------
# --- BUILDER STAGE ---
# ---------------------
FROM python:3.13-slim AS builder

WORKDIR /app

# Install only what's needed for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build wheels with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_BUILD_ISOLATION=false \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt


# -------------------
# --- FINAL STAGE ---
# -------------------
FROM python:3.13-slim

WORKDIR /app

# Only git, keep base as clean as possible
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels \
    && rm -rf /root/.cache


COPY . .


EXPOSE 8686

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8686/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8686"]
