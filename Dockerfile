# Base image with Python 3.13
FROM python:3.12-slim AS base

# Recommended environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Intall the requiered dependencies of the system 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv and the dependencies of the project
FROM base AS builder
# Install uv
RUN pip install --no-cache-dir uv
# Copy the dependencies file
COPY pyproject.toml .
# Copy the lock file if exists
COPY uv.lock* .
# Install the dependencies of the project in the system's environment
RUN uv pip install --system --no-cache .

# Copy the source code and prepare the execution environment
FROM base AS runtime
# Copy the installed dependencies
COPY --from=builder /usr/local /usr/local
# Copy your application code
COPY api/ ./api/
COPY logic/ ./logic/

COPY artifacts/ ./artifacts/
COPY data/ ./data/

# Expose the port associated with the API created with FastAPI
EXPOSE 8000
# Default command: it starts the API with uvicorn
CMD ["sh", "-c", "uvicorn api.api:app --host 0.0.0.0 --port ${PORT}"]
