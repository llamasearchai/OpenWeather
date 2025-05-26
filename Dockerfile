# OpenWeather Platform Dockerfile
# Multi-stage build for production optimization

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add labels
LABEL maintainer="OpenWeather Team <team@openweather.com>"
LABEL org.opencontainers.image.title="OpenWeather Platform"
LABEL org.opencontainers.image.description="Enterprise-grade weather analytics platform with AI, LLM integration, and drone support"
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.url="https://github.com/openweather/platform"
LABEL org.opencontainers.image.source="https://github.com/openweather/platform"

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt || pip install \
        fastapi[all]==0.104.1 \
        uvicorn[standard]==0.24.0 \
        python-multipart==0.0.6 \
        python-jose[cryptography]==3.3.0 \
        passlib[bcrypt]==1.7.4 \
        redis==5.0.1 \
        httpx==0.25.2 \
        pydantic==2.5.0 \
        jinja2==3.1.2 \
        aiofiles==23.2.1 \
        prometheus-client==0.19.0 \
        psutil==5.9.6 \
        plotly==5.17.0 \
        pandas==2.1.4 \
        numpy==1.24.4 \
        scikit-learn==1.3.2 \
        pymavlink==2.4.37 \
        openai==1.3.8 \
        anthropic==0.7.8

# Production stage
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=info \
    HOST=0.0.0.0 \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r openweather && \
    useradd -r -g openweather -d /app -s /bin/bash openweather && \
    mkdir -p /app/logs /app/data /app/cache && \
    chown -R openweather:openweather /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=openweather:openweather . .

# Create necessary directories
RUN mkdir -p \
    /app/openweather/web/static \
    /app/openweather/web/templates \
    /app/logs \
    /app/data \
    /app/cache && \
    chown -R openweather:openweather /app

# Switch to non-root user
USER openweather

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "-m", "openweather.main"]

# Default command
CMD [] 