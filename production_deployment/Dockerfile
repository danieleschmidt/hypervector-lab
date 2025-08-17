# Production Dockerfile for HyperVector-Lab
FROM python:3.11-slim as builder

# Set build arguments
ARG CUDA_VERSION=12.1
ARG PYTHON_VERSION=3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch numpy scipy scikit-learn \
    && pip install --no-cache-dir matplotlib tqdm einops transformers pillow

# Production stage
FROM python:3.11-slim as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r hypervector && useradd -r -g hypervector hypervector

# Set working directory
WORKDIR /app

# Copy application code
COPY hypervector/ ./hypervector/
COPY pyproject.toml ./
COPY README.md ./

# Set proper permissions
RUN chown -R hypervector:hypervector /app
USER hypervector

# Expose port for API if needed
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hypervector; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import hypervector; print('HyperVector-Lab ready')"]
