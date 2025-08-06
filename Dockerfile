FROM python:3.11-slim

LABEL maintainer="HyperVector Lab <contact@hyperdimensional.co>"
LABEL description="Production-ready Hyperdimensional Computing in PyTorch"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app
ENV HYPERVECTOR_DEVICE=cpu
ENV HYPERVECTOR_LOG_LEVEL=INFO

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Copy source code
COPY hypervector/ ./hypervector/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Run basic validation
RUN python -c "import hypervector; print('HyperVector-Lab installed successfully')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hypervector; hv = hypervector.HDCSystem(dim=100); print('OK')" || exit 1

# Expose port for API (if using FastAPI/Flask)
EXPOSE 8000

# Default command
CMD ["python", "-m", "hypervector.benchmark", "--basic"]