#!/bin/bash
set -e

echo "🚀 Starting HyperVector-Lab Production Deployment..."

# Configuration
NAMESPACE="hypervector-prod"
IMAGE_TAG="hypervector-lab:$(date +%Y%m%d-%H%M%S)"

# Functions
log_info() {
    echo "[INFO] $(date): $1"
}

log_error() {
    echo "[ERROR] $(date): $1" >&2
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

build_image() {
    log_info "Building Docker image..."
    if [ -f "Dockerfile.production" ]; then
        docker build -f Dockerfile.production -t $IMAGE_TAG .
        log_info "Docker image built: $IMAGE_TAG"
    else
        log_error "Dockerfile.production not found"
        exit 1
    fi
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Test the container
    docker run --rm $IMAGE_TAG python -c "
import hypervector
hdc = hypervector.HDCSystem(dim=100)
print('✅ HyperVector system initialized successfully')
"
    
    log_info "Health checks completed"
}

main() {
    log_info "Starting deployment process..."
    
    check_prerequisites
    build_image
    run_health_checks
    
    log_info "🎉 Deployment completed successfully!"
    log_info "Image ready: $IMAGE_TAG"
}

# Error handling
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Run deployment
main "$@"
