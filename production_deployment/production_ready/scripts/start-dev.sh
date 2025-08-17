#!/bin/bash
set -euo pipefail

echo "Starting HyperVector-Lab local development environment..."

# Start services using Docker Compose
cd ../docker
docker-compose up -d

echo "Services starting..."
echo "HyperVector-Lab: http://localhost:8000"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Test connectivity
if curl -f http://localhost:8000 2>/dev/null; then
    echo "✓ HyperVector-Lab is ready"
else
    echo "⚠ HyperVector-Lab may still be starting"
fi

echo "Development environment ready!"
echo "Run 'docker-compose logs -f' to see logs"
echo "Run 'docker-compose down' to stop services"
