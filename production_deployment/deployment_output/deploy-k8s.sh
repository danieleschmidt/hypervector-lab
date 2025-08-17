#!/bin/bash
set -e

echo "🚀 Starting Kubernetes deployment..."

# Apply ConfigMap
echo "⚙️  Applying ConfigMap..."
kubectl apply -f configmap.yaml

# Apply Deployment
echo "🚀 Applying Deployment..."
kubectl apply -f deployment.yaml

# Apply Service
echo "🌐 Applying Service..."
kubectl apply -f service.yaml

echo "✅ Kubernetes deployment completed!"
echo "📊 Check status with: kubectl get pods,services"
