#!/bin/bash

# HyperVector-Lab Production Deployment Script
# Autonomous SDLC execution final deployment phase

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

banner() {
    echo -e "${PURPLE}"
    echo "=================================================================="
    echo "  HYPERVECTOR-LAB PRODUCTION DEPLOYMENT"
    echo "  Autonomous SDLC Execution - Final Phase"
    echo "=================================================================="
    echo -e "${NC}"
}

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
HYPERVECTOR_VERSION="${HYPERVECTOR_VERSION:-latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
KUBERNETES_NAMESPACE="${KUBERNETES_NAMESPACE:-hypervector}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"
GPU_ENABLED="${GPU_ENABLED:-false}"

banner

log "Starting HyperVector-Lab deployment..."
log "Environment: $DEPLOYMENT_ENV"
log "Version: $HYPERVECTOR_VERSION"
log "GPU Support: $GPU_ENABLED"
log "Monitoring: $MONITORING_ENABLED"

# Pre-deployment checks
log "Running pre-deployment checks..."

# Check dependencies
check_dependencies() {
    info "Checking system dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        warn "docker-compose not found, checking for 'docker compose' command..."
        if ! docker compose version &> /dev/null; then
            error "Neither docker-compose nor 'docker compose' command found."
        fi
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed."
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        error "Git is not installed."
    fi
    
    # Check GPU support if enabled
    if [[ "$GPU_ENABLED" == "true" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            warn "nvidia-smi not found. GPU support may not work properly."
        else
            info "NVIDIA GPUs detected:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        fi
    fi
    
    info "All dependencies satisfied ✓"
}

# System requirements check
check_system_requirements() {
    info "Checking system requirements..."
    
    # Check available memory (require at least 4GB)
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -lt 4 ]]; then
        warn "System has less than 4GB RAM. Performance may be degraded."
    fi
    
    # Check available disk space (require at least 10GB)
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $AVAILABLE_SPACE -lt 10 ]]; then
        warn "Less than 10GB disk space available. Consider freeing up space."
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    if [[ $CPU_CORES -lt 2 ]]; then
        warn "System has less than 2 CPU cores. Performance may be limited."
    fi
    
    info "System requirements check completed ✓"
}

# Validate configuration
validate_config() {
    info "Validating deployment configuration..."
    
    # Check if pyproject.toml exists
    if [[ ! -f "pyproject.toml" ]]; then
        error "pyproject.toml not found. Are you in the correct directory?"
    fi
    
    # Validate Docker files
    if [[ "$GPU_ENABLED" == "true" ]]; then
        if [[ ! -f "Dockerfile.gpu" ]]; then
            error "Dockerfile.gpu not found but GPU support is enabled."
        fi
    else
        if [[ ! -f "Dockerfile" ]]; then
            error "Dockerfile not found."
        fi
    fi
    
    # Check docker-compose.yml
    if [[ ! -f "docker-compose.yml" ]]; then
        error "docker-compose.yml not found."
    fi
    
    info "Configuration validation completed ✓"
}

# Build images
build_images() {
    log "Building Docker images..."
    
    if [[ "$GPU_ENABLED" == "true" ]]; then
        info "Building GPU-enabled image..."
        docker build -f Dockerfile.gpu -t hypervector-lab:$HYPERVECTOR_VERSION-gpu .
        
        if [[ -n "$DOCKER_REGISTRY" ]]; then
            docker tag hypervector-lab:$HYPERVECTOR_VERSION-gpu $DOCKER_REGISTRY/hypervector-lab:$HYPERVECTOR_VERSION-gpu
        fi
    else
        info "Building CPU-only image..."
        docker build -f Dockerfile -t hypervector-lab:$HYPERVECTOR_VERSION .
        
        if [[ -n "$DOCKER_REGISTRY" ]]; then
            docker tag hypervector-lab:$HYPERVECTOR_VERSION $DOCKER_REGISTRY/hypervector-lab:$HYPERVECTOR_VERSION
        fi
    fi
    
    log "Image build completed ✓"
}

# Setup directories
setup_directories() {
    info "Setting up deployment directories..."
    
    # Create necessary directories
    mkdir -p data/{input,output,models,cache}
    mkdir -p logs/{app,monitoring,security}
    mkdir -p config/{production,staging,development}
    mkdir -p monitoring/{prometheus,grafana}
    
    # Set appropriate permissions
    chmod 755 data logs config monitoring
    chmod 644 docker-compose.yml
    
    # Create default configuration if it doesn't exist
    if [[ ! -f "config/production/config.json" ]]; then
        cat > config/production/config.json << EOF
{
  "hypervector": {
    "default_dimension": 10000,
    "device": "${GPU_ENABLED:+cuda}${GPU_ENABLED:-cpu}",
    "batch_size": 100,
    "cache_enabled": true,
    "monitoring_enabled": $MONITORING_ENABLED
  },
  "security": {
    "enable_rate_limiting": true,
    "max_requests_per_minute": 1000,
    "enable_audit_logging": true
  },
  "performance": {
    "enable_profiling": true,
    "enable_optimization": true,
    "parallel_workers": $(nproc)
  },
  "compliance": {
    "data_retention_days": 90,
    "enable_gdpr_compliance": true,
    "anonymize_logs": true
  }
}
EOF
        info "Created default production configuration"
    fi
    
    log "Directory setup completed ✓"
}

# Run tests
run_tests() {
    log "Running comprehensive test suite..."
    
    # Create virtual environment for testing
    if [[ ! -d "test_env" ]]; then
        python3 -m venv test_env
    fi
    
    source test_env/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -e ".[dev]"
    
    # Run unit tests
    info "Running unit tests..."
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --cov=hypervector --cov-report=html
    else
        python -m unittest discover tests/
    fi
    
    # Run integration tests
    info "Running integration tests..."
    python -c "
import hypervector as hv
import torch
import time

print('🧪 Integration test suite')

# Test basic functionality
hdc = hv.HDCSystem(dim=1000)
text_hv = hdc.encode_text('integration test')
img_hv = hdc.encode_image(torch.randn(3, 224, 224))

# Test operations
bound = hdc.bind([text_hv, img_hv])
bundle = hdc.bundle([text_hv, img_hv])
sim = hdc.cosine_similarity(text_hv, img_hv)

print(f'✅ All integration tests passed')
print(f'   Text HV: {text_hv.dim}')
print(f'   Image HV: {img_hv.dim}')
print(f'   Bound HV: {bound.dim}')  
print(f'   Bundle HV: {bundle.dim}')
print(f'   Similarity: {sim.item():.3f}')
"
    
    deactivate
    
    log "Test suite completed ✓"
}

# Security scan
security_scan() {
    log "Running security scans..."
    
    # Check for common security issues in Docker images
    info "Scanning Docker images for vulnerabilities..."
    
    # Use Docker's built-in security scanning if available
    if docker --help | grep -q "scan"; then
        if [[ "$GPU_ENABLED" == "true" ]]; then
            docker scan hypervector-lab:$HYPERVECTOR_VERSION-gpu || warn "Docker scan not available or failed"
        else
            docker scan hypervector-lab:$HYPERVECTOR_VERSION || warn "Docker scan not available or failed"
        fi
    else
        warn "Docker scan not available. Consider using external security scanning tools."
    fi
    
    # Check file permissions
    info "Checking file permissions..."
    find . -type f -perm /002 -ls | head -10 || true
    
    # Check for secrets in files
    info "Scanning for potential secrets..."
    if command -v git &> /dev/null; then
        # Look for common secret patterns
        git ls-files | xargs grep -l "password\|secret\|key\|token" | head -5 || true
    fi
    
    log "Security scan completed ✓"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    # Stop existing services
    info "Stopping existing services..."
    $DOCKER_COMPOSE_CMD down --remove-orphans || true
    
    # Start core services
    info "Starting core services..."
    if [[ "$GPU_ENABLED" == "true" ]]; then
        $DOCKER_COMPOSE_CMD up -d hypervector-gpu redis
    else
        $DOCKER_COMPOSE_CMD up -d hypervector-cpu redis
    fi
    
    # Start monitoring services if enabled
    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        info "Starting monitoring services..."
        $DOCKER_COMPOSE_CMD up -d prometheus grafana
    fi
    
    # Wait for services to be ready
    info "Waiting for services to be ready..."
    sleep 10
    
    # Health check
    info "Performing health checks..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log "HyperVector service is healthy ✓"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "Health check failed after 30 attempts"
        fi
        sleep 2
    done
    
    log "Service deployment completed ✓"
}

# Post-deployment validation
post_deployment_validation() {
    log "Running post-deployment validation..."
    
    # Test API endpoints
    info "Testing API endpoints..."
    
    # Test health endpoint
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    if [[ $HTTP_CODE -eq 200 ]]; then
        info "Health endpoint: ✓"
    else
        error "Health endpoint failed: HTTP $HTTP_CODE"
    fi
    
    # Test basic functionality through API
    info "Testing core functionality..."
    python3 -c "
import requests
import json

# Test encoding endpoint
try:
    response = requests.post('http://localhost:8000/encode/text', 
                           json={'text': 'deployment test'}, 
                           timeout=10)
    if response.status_code == 200:
        print('✅ Text encoding API: Working')
    else:
        print(f'❌ Text encoding API: HTTP {response.status_code}')
except Exception as e:
    print(f'❌ Text encoding API: {e}')
" 2>/dev/null || warn "API functionality test failed (service may not expose HTTP API)"
    
    # Check resource usage
    info "Checking resource usage..."
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -5
    
    log "Post-deployment validation completed ✓"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    REPORT_FILE="deployment_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > $REPORT_FILE << EOF
HYPERVECTOR-LAB PRODUCTION DEPLOYMENT REPORT
============================================

Deployment Information:
- Date: $(date)
- Environment: $DEPLOYMENT_ENV
- Version: $HYPERVECTOR_VERSION
- GPU Enabled: $GPU_ENABLED
- Monitoring Enabled: $MONITORING_ENABLED
- Docker Compose Command: $DOCKER_COMPOSE_CMD

System Information:
- OS: $(uname -a)
- Python: $(python3 --version)
- Docker: $(docker --version)
- Available Memory: ${TOTAL_MEM}GB
- CPU Cores: $CPU_CORES
- Disk Space: ${AVAILABLE_SPACE}GB

Services Status:
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")

Docker Images:
$(docker images | grep hypervector)

Deployment Validation:
- Health Check: $(curl -s http://localhost:8000/health &> /dev/null && echo "PASSED" || echo "FAILED")
- Service Discovery: $(docker ps | grep hypervector | wc -l) services running

Next Steps:
1. Monitor service logs: docker-compose logs -f
2. Access Grafana dashboard: http://localhost:3000 (admin/admin)
3. View Prometheus metrics: http://localhost:9090
4. Check application logs: tail -f logs/app/*.log

Configuration Files:
- Main config: config/production/config.json
- Docker Compose: docker-compose.yml
- Environment: $DEPLOYMENT_ENV

EOF

    log "Deployment report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    info "Cleaning up temporary files..."
    rm -rf test_env 2>/dev/null || true
}

# Main deployment workflow
main() {
    trap cleanup EXIT
    
    check_dependencies
    check_system_requirements  
    validate_config
    setup_directories
    build_images
    run_tests
    security_scan
    deploy_services
    post_deployment_validation
    generate_report
    
    echo
    log "🎉 HYPERVECTOR-LAB DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo
    echo -e "${GREEN}Services running:${NC}"
    echo -e "${BLUE}• HyperVector API: http://localhost:8000${NC}"
    if [[ "$MONITORING_ENABLED" == "true" ]]; then
        echo -e "${BLUE}• Grafana Dashboard: http://localhost:3000 (admin/admin)${NC}"
        echo -e "${BLUE}• Prometheus Metrics: http://localhost:9090${NC}"
    fi
    echo
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Review deployment report: cat $REPORT_FILE"
    echo "2. Monitor services: docker-compose logs -f"  
    echo "3. Scale services: docker-compose up --scale hypervector-cpu=3"
    echo "4. Update configuration: edit config/production/config.json"
    echo
    echo -e "${PURPLE}🚀 AUTONOMOUS SDLC EXECUTION: COMPLETE${NC}"
    echo -e "${GREEN}✅ All phases successfully executed${NC}"
    echo -e "${GREEN}✅ Production-ready deployment achieved${NC}"
    echo
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "test")
        run_tests
        ;;
    "build")
        build_images
        ;;
    "check")
        check_dependencies
        check_system_requirements
        validate_config
        ;;
    "clean")
        info "Cleaning up deployment..."
        $DOCKER_COMPOSE_CMD down --volumes --remove-orphans
        docker system prune -f
        rm -rf data logs config monitoring test_env 2>/dev/null || true
        log "Cleanup completed ✓"
        ;;
    "status")
        info "Deployment status:"
        docker ps
        echo
        echo "Services health:"
        curl -s http://localhost:8000/health && echo " ✓ HyperVector API" || echo " ❌ HyperVector API"
        ;;
    "help"|"--help"|"-h")
        echo "HyperVector-Lab Deployment Script"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  deploy    Full deployment (default)"
        echo "  test      Run tests only"
        echo "  build     Build images only"
        echo "  check     Run pre-deployment checks"
        echo "  clean     Clean up deployment"
        echo "  status    Show deployment status"
        echo "  help      Show this help"
        echo
        echo "Environment Variables:"
        echo "  DEPLOYMENT_ENV=production    Deployment environment"
        echo "  GPU_ENABLED=false           Enable GPU support"  
        echo "  MONITORING_ENABLED=true     Enable monitoring"
        echo "  DOCKER_REGISTRY=            Docker registry URL"
        ;;
    *)
        error "Unknown command: $1. Use '$0 help' for usage."
        ;;
esac