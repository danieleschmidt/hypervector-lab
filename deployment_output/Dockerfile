# Multi-stage build for HyperVector-Lab production deployment
FROM python:3.11-slim AS builder

# Set build arguments
ARG BUILD_ENV=development
ARG VERSION=1.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r hypervector && useradd -r -g hypervector hypervector

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HYPERVECTOR_ENV=development

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache

# Change ownership to non-root user
RUN chown -R hypervector:hypervector /app

# Switch to non-root user
USER hypervector

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python", "-m", "hypervector.server", "--host", "0.0.0.0", "--port", "8000"]
