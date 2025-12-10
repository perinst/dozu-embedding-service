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
    pip wheel --no-deps --wheel-dir=/wheels -r requirements.txt


# -------------------
# --- FINAL STAGE ---
# -------------------
FROM python:3.13-slim

WORKDIR /app

# Install git (minimal runtime deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY src/ ./src/
COPY *.py ./


RUN rm -rf /root/.cache/pip

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser


ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8686

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8686/health')" || exit 1

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8686"]
