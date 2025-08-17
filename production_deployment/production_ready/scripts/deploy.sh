#!/bin/bash
set -euo pipefail

# HyperVector-Lab Deployment Script
ENVIRONMENT=${1:-development}
NAMESPACE=${2:-hypervector-lab}

echo "Deploying HyperVector-Lab to $ENVIRONMENT environment..."

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations
kubectl apply -f ../kubernetes/configmap.yaml -n $NAMESPACE

# Deploy application
kubectl apply -f ../kubernetes/deployment.yaml -n $NAMESPACE
kubectl apply -f ../kubernetes/service.yaml -n $NAMESPACE
kubectl apply -f ../kubernetes/hpa.yaml -n $NAMESPACE

# Wait for deployment to be ready
kubectl rollout status deployment/hypervector-lab -n $NAMESPACE --timeout=300s

echo "Deployment completed successfully!"

# Show status
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
