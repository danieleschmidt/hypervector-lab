#!/bin/bash

# HyperVector-Lab Deployment Script
# Environment: development
# Target: docker_compose

set -euo pipefail

APP_NAME="hypervector-api"
ENVIRONMENT="development"
DOCKER_IMAGE="hypervector-lab:latest"

echo "🚀 Starting HyperVector-Lab deployment"
echo "Environment: $ENVIRONMENT"
echo "Target: docker_compose"

# Pre-deployment checks
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker daemon is not running"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Build Docker image
build_image() {
    echo "🔨 Building Docker image..."
    
    docker build \
        --tag "$DOCKER_IMAGE" \
        --build-arg BUILD_ENV="$ENVIRONMENT" \
        .
    
    echo "✅ Docker image built successfully"
}

# Deploy based on target platform
deploy() {
    echo "📦 Starting deployment..."
    
    case "docker_compose" in
        "docker_compose")
            deploy_docker_compose
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        *)
            echo "❌ Unknown deployment target: docker_compose"
            exit 1
            ;;
    esac
}

# Deploy using Docker Compose
deploy_docker_compose() {
    echo "🐳 Deploying using Docker Compose..."
    
    if [ ! -f "docker-compose.yml" ]; then
        echo "❌ docker-compose.yml not found"
        exit 1
    fi
    
    docker-compose down || true
    docker-compose up -d
    
    echo "✅ Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo "☸️  Deploying to Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl is not installed"
        exit 1
    fi
    
    kubectl apply -f configmap.yaml
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    
    kubectl rollout status deployment/"$APP_NAME" --timeout=600s
    
    echo "✅ Kubernetes deployment completed"
}

# Wait for health check
wait_for_health() {
    echo "🏥 Waiting for application to become healthy..."
    
    for i in {1..30}; do
        if curl -f -s "http://localhost:8000/health" > /dev/null 2>&1; then
            echo "✅ Application is healthy"
            return 0
        fi
        echo "⏳ Health check attempt $i/30, retrying in 10 seconds..."
        sleep 10
    done
    
    echo "❌ Application failed to become healthy"
    return 1
}

# Run smoke tests
run_smoke_tests() {
    echo "🧪 Running smoke tests..."
    
    if curl -f -s "http://localhost:8000/health" | grep -q "healthy"; then
        echo "✅ Health endpoint test passed"
    else
        echo "❌ Health endpoint test failed"
        return 1
    fi
    
    echo "✅ All smoke tests passed"
}

# Main deployment flow
main() {
    check_prerequisites
    build_image
    deploy
    wait_for_health
    run_smoke_tests
    
    echo "🎉 Deployment completed successfully!"
    echo "Application is available at: http://localhost:8000"
}

main
