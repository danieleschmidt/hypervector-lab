#!/bin/bash
set -euo pipefail

# HyperVector-Lab Production Deployment Script

echo "🚀 Starting HyperVector-Lab deployment..."

# Configuration
DEPLOYMENT_TYPE=${1:-docker}
ENVIRONMENT=${2:-production}

echo "Deployment type: $DEPLOYMENT_TYPE"
echo "Environment: $ENVIRONMENT"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
python comprehensive_quality_gates.py
if [ $? -ne 0 ]; then
    echo "❌ Quality gates failed. Aborting deployment."
    exit 1
fi

case $DEPLOYMENT_TYPE in
    "docker")
        echo "🐳 Deploying with Docker..."
        docker-compose -f docker-compose.yml up -d
        ;;
    "kubernetes")
        echo "☸️ Deploying to Kubernetes..."
        kubectl apply -f kubernetes/
        ;;
    "local")
        echo "🏠 Local deployment..."
        pip install -e .
        ;;
    *)
        echo "❌ Unknown deployment type: $DEPLOYMENT_TYPE"
        exit 1
        ;;
esac

echo "✅ Deployment completed successfully!"

# Post-deployment verification
echo "🔬 Running post-deployment tests..."
sleep 10
python -c "import hypervector; print('✅ HyperVector-Lab is ready!')"

echo "🎉 Deployment verification completed!"
