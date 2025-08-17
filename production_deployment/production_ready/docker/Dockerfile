# HyperVector-Lab Production Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY hypervector/ ./hypervector/
COPY examples/ ./examples/
COPY README.md LICENSE ./

# Create non-root user
RUN useradd -m -u 1000 hypervector && \
    chown -R hypervector:hypervector /app
USER hypervector

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hypervector; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import hypervector; print('HyperVector-Lab ready for production')"]

EXPOSE 8000
