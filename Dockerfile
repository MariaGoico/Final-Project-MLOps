# Base image with Python 3.12
FROM python:3.12-slim AS base

# Recommended environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv and dependencies
FROM base AS builder
RUN pip install --no-cache-dir uv

# Copy dependency definition
COPY pyproject.toml .
# Copy lock file if exists
COPY uv.lock* .

RUN uv pip install --system --no-cache .

# Prepare runtime environment
FROM base AS runtime

# Copy libraries from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY api/ ./api/
COPY logic/ ./logic/

# Copy artifacts (Modelos generados por GitHub Actions)
COPY artifacts/ ./artifacts/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.api:app --host 0.0.0.0 --port ${PORT}"]