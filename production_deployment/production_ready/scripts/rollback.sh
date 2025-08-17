#!/bin/bash
set -euo pipefail

NAMESPACE=${1:-hypervector-lab}
DEPLOYMENT=${2:-hypervector-lab}

echo "Rolling back HyperVector-Lab deployment..."

# Rollback to previous version
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollback to complete
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

echo "Rollback completed successfully!"

# Show current status
kubectl get pods -n $NAMESPACE
kubectl describe deployment/$DEPLOYMENT -n $NAMESPACE
