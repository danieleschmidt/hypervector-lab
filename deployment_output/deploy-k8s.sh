#!/bin/bash

# Kubernetes Deployment Script for HyperVector-Lab

set -euo pipefail

APP_NAME="hypervector-api"

echo "☸️  Deploying to Kubernetes..."

# Apply manifests
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
kubectl wait --for=condition=available --timeout=600s deployment/$APP_NAME

echo "✅ Deployment completed successfully!"

# Show status
kubectl get pods -l app=hypervector
kubectl get svc -l app=hypervector
