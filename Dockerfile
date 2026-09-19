FROM python:3.12-slim

# Install system dependencies (including ca-certificates for secure connections)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY src/ ./src/
COPY README.md ./

# Sync project package itself
RUN uv sync --frozen --no-dev

# Set Python environment path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Download and pre-cache model weights (e.g. Silero VAD) into the container image
RUN python -m livekit.agents download-files

# Run the LiveKit Agent worker
CMD ["python", "-m", "zcloud_ai_telecaller.agent", "start"]

