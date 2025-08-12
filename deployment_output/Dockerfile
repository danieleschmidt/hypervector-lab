FROM python:3.12-slim

LABEL maintainer="HyperVector Lab Team"
LABEL description="HyperVector-Lab Production Deployment"
LABEL version="1.0.0"

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    libgomp1
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Copy application code
COPY . .

# Install HyperVector package
RUN pip install -e .

# Environment variables
ENV HYPERVECTOR_LOG_LEVEL=INFO
ENV HYPERVECTOR_CACHE_SIZE=1000
ENV HYPERVECTOR_MAX_MEMORY_GB=2
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import hypervector; print('Health check passed')" || exit 1

# Create non-root user
RUN groupadd -g 1000 appuser && useradd -r -u 1000 -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "-c", "import hypervector; print('HyperVector container started successfully')"]