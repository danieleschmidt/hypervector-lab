#!/bin/bash
set -euo pipefail

NAMESPACE=${1:-hypervector-lab}
SERVICE_NAME=${2:-hypervector-service}

echo "Checking health of HyperVector-Lab deployment..."

# Check pod status
echo "Pod Status:"
kubectl get pods -n $NAMESPACE

# Check service status
echo "Service Status:"
kubectl get services -n $NAMESPACE

# Check deployment status
echo "Deployment Status:"
kubectl get deployments -n $NAMESPACE

# Test service connectivity
echo "Testing service connectivity..."
kubectl port-forward service/$SERVICE_NAME 8080:8000 -n $NAMESPACE &
PF_PID=$!

sleep 5

# Test health endpoint (if exists)
if curl -f http://localhost:8080/health 2>/dev/null; then
    echo "✓ Health check passed"
else
    echo "⚠ Health check endpoint not available"
fi

# Kill port-forward
kill $PF_PID 2>/dev/null || true

echo "Health check completed!"
