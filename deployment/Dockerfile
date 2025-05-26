# Multi-stage Dockerfile for OpenWeather Platform
# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root

# Stage 2: Runtime environment
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r openweather && \
    useradd -r -g openweather -d /app -s /bin/bash openweather

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=openweather:openweather . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/tmp && \
    chown -R openweather:openweather /app

# Switch to non-root user
USER openweather

# Install the application
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "openweather.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Labels for metadata
LABEL maintainer="OpenWeather Team" \
      version="1.0.0" \
      description="OpenWeather Platform - Advanced Weather Analytics with AI" \
      org.opencontainers.image.source="https://github.com/your-org/openweather" \
      org.opencontainers.image.documentation="https://docs.openweather.com" \
      org.opencontainers.image.licenses="MIT"