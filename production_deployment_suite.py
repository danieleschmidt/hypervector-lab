#!/usr/bin/env python3
"""Production deployment suite for HyperVector-Lab."""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil


class ProductionDeploymentSuite:
    """Complete production deployment preparation."""
    
    def __init__(self):
        self.repo_root = Path('/root/repo')
        self.deployment_dir = self.repo_root / 'production_ready'
        
    def prepare_complete_deployment(self) -> bool:
        """Prepare complete production deployment."""
        print("=" * 80)
        print("HYPERVECTOR-LAB PRODUCTION DEPLOYMENT SUITE")
        print("=" * 80)
        
        steps = [
            ("Validate Environment", self._validate_environment),
            ("Create Deployment Structure", self._create_deployment_structure),
            ("Generate Docker Assets", self._generate_docker_assets),
            ("Create Kubernetes Manifests", self._create_kubernetes_manifests),
            ("Setup Monitoring Stack", self._setup_monitoring_stack),
            ("Create CI/CD Pipeline", self._create_cicd_pipeline),
            ("Generate Environment Configs", self._generate_environment_configs),
            ("Create Deployment Scripts", self._create_deployment_scripts),
            ("Validate Deployment", self._validate_deployment)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}:")
            print("-" * 50)
            
            try:
                success = step_func()
                if success:
                    print("✓ SUCCESS")
                else:
                    print("✗ FAILED")
                    return False
            except Exception as e:
                print(f"✗ ERROR: {e}")
                return False
        
        print("\n" + "=" * 80)
        print("✓ PRODUCTION DEPLOYMENT SUITE COMPLETE")
        print(f"Deployment artifacts ready in: {self.deployment_dir}")
        print("=" * 80)
        
        return True
    
    def _validate_environment(self) -> bool:
        """Validate deployment environment."""
        # Check required directories exist
        required_dirs = [
            'hypervector',
            'tests',
            'examples'
        ]
        
        for dir_name in required_dirs:
            if not (self.repo_root / dir_name).exists():
                print(f"✗ Missing required directory: {dir_name}")
                return False
            print(f"✓ Found directory: {dir_name}")
        
        # Check required files
        required_files = [
            'pyproject.toml',
            'README.md',
            'LICENSE'
        ]
        
        for file_name in required_files:
            if not (self.repo_root / file_name).exists():
                print(f"✗ Missing required file: {file_name}")
                return False
            print(f"✓ Found file: {file_name}")
        
        return True
    
    def _create_deployment_structure(self) -> bool:
        """Create production deployment directory structure."""
        # Create main deployment directory
        self.deployment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            'docker',
            'kubernetes',
            'monitoring',
            'ci-cd',
            'configs',
            'scripts',
            'docs'
        ]
        
        for subdir in subdirs:
            (self.deployment_dir / subdir).mkdir(exist_ok=True)
            print(f"✓ Created directory: {subdir}")
        
        return True
    
    def _generate_docker_assets(self) -> bool:
        """Generate comprehensive Docker assets."""
        docker_dir = self.deployment_dir / 'docker'
        
        # Production Dockerfile
        dockerfile_content = '''# HyperVector-Lab Production Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY hypervector/ ./hypervector/
COPY examples/ ./examples/
COPY README.md LICENSE ./

# Create non-root user
RUN useradd -m -u 1000 hypervector && \\
    chown -R hypervector:hypervector /app
USER hypervector

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import hypervector; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import hypervector; print('HyperVector-Lab ready for production')"]

EXPOSE 8000
'''
        
        with open(docker_dir / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        print("✓ Generated production Dockerfile")
        
        # Docker Compose for development
        compose_content = '''version: '3.8'

services:
  hypervector-lab:
    build:
      context: ../..
      dockerfile: production_ready/docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
    volumes:
      - ../../examples:/app/examples:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import hypervector; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml:ro
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
'''
        
        with open(docker_dir / 'docker-compose.yml', 'w') as f:
            f.write(compose_content)
        print("✓ Generated Docker Compose configuration")
        
        # Multi-stage build for GPU support
        gpu_dockerfile = '''# HyperVector-Lab GPU Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3.11-dev \\
    python3-pip \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[cuda]"

FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS production

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3-pip \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY hypervector/ ./hypervector/
COPY examples/ ./examples/

RUN useradd -m -u 1000 hypervector && \\
    chown -R hypervector:hypervector /app
USER hypervector

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

CMD ["python3", "-c", "import hypervector; print('HyperVector-Lab GPU ready')"]
'''
        
        with open(docker_dir / 'Dockerfile.gpu', 'w') as f:
            f.write(gpu_dockerfile)
        print("✓ Generated GPU Dockerfile")
        
        return True
    
    def _create_kubernetes_manifests(self) -> bool:
        """Create Kubernetes deployment manifests."""
        k8s_dir = self.deployment_dir / 'kubernetes'
        
        # Namespace
        namespace_manifest = '''apiVersion: v1
kind: Namespace
metadata:
  name: hypervector-lab
  labels:
    name: hypervector-lab
'''
        
        with open(k8s_dir / 'namespace.yaml', 'w') as f:
            f.write(namespace_manifest)
        
        # ConfigMap
        configmap_manifest = '''apiVersion: v1
kind: ConfigMap
metadata:
  name: hypervector-config
  namespace: hypervector-lab
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  MAX_WORKERS: "4"
  CACHE_TTL: "3600"
'''
        
        with open(k8s_dir / 'configmap.yaml', 'w') as f:
            f.write(configmap_manifest)
        
        # Deployment
        deployment_manifest = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypervector-lab
  namespace: hypervector-lab
  labels:
    app: hypervector-lab
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: hypervector-lab
  template:
    metadata:
      labels:
        app: hypervector-lab
    spec:
      containers:
      - name: hypervector-lab
        image: hypervector-lab:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: hypervector-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import hypervector; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import hypervector; print('OK')"
          initialDelaySeconds: 5
          periodSeconds: 10
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
      securityContext:
        fsGroup: 1000
'''
        
        with open(k8s_dir / 'deployment.yaml', 'w') as f:
            f.write(deployment_manifest)
        
        # Service
        service_manifest = '''apiVersion: v1
kind: Service
metadata:
  name: hypervector-service
  namespace: hypervector-lab
  labels:
    app: hypervector-lab
spec:
  selector:
    app: hypervector-lab
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
'''
        
        with open(k8s_dir / 'service.yaml', 'w') as f:
            f.write(service_manifest)
        
        # HorizontalPodAutoscaler
        hpa_manifest = '''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hypervector-hpa
  namespace: hypervector-lab
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hypervector-lab
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
        
        with open(k8s_dir / 'hpa.yaml', 'w') as f:
            f.write(hpa_manifest)
        
        print("✓ Generated Kubernetes manifests")
        return True
    
    def _setup_monitoring_stack(self) -> bool:
        """Setup comprehensive monitoring stack."""
        monitoring_dir = self.deployment_dir / 'monitoring'
        
        # Prometheus configuration
        prometheus_config = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'hypervector-lab'
    static_configs:
      - targets: ['hypervector-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
'''
        
        with open(monitoring_dir / 'prometheus.yml', 'w') as f:
            f.write(prometheus_config)
        
        # Grafana datasources
        grafana_datasources = '''apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
'''
        
        with open(monitoring_dir / 'grafana-datasources.yml', 'w') as f:
            f.write(grafana_datasources)
        
        # Alert rules
        alert_rules = '''groups:
  - name: hypervector-lab-alerts
    rules:
    - alert: HighCPUUsage
      expr: cpu_usage > 80
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High CPU usage detected"
        description: "CPU usage is above 80% for more than 5 minutes"
        
    - alert: HighMemoryUsage
      expr: memory_usage > 85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage detected"
        description: "Memory usage is above 85% for more than 5 minutes"
        
    - alert: ServiceDown
      expr: up{job="hypervector-lab"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "HyperVector-Lab service is down"
        description: "HyperVector-Lab service has been down for more than 1 minute"
'''
        
        with open(monitoring_dir / 'alert_rules.yml', 'w') as f:
            f.write(alert_rules)
        
        print("✓ Generated monitoring configuration")
        return True
    
    def _create_cicd_pipeline(self) -> bool:
        """Create CI/CD pipeline configurations."""
        cicd_dir = self.deployment_dir / 'ci-cd'
        
        # GitHub Actions workflow
        github_workflow = '''name: HyperVector-Lab CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: hypervector-lab

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run quality gates
      run: |
        python quality_gates_validator.py
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=hypervector
    
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: production_ready/docker/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      run: |
        echo "Deployment would happen here"
        # kubectl apply -f production_ready/kubernetes/
'''
        
        with open(cicd_dir / 'github-actions.yml', 'w') as f:
            f.write(github_workflow)
        
        # GitLab CI configuration
        gitlab_ci = '''stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  KUBE_NAMESPACE: hypervector-lab

before_script:
  - python -m pip install --upgrade pip

test:
  stage: test
  image: python:3.11
  script:
    - pip install -e ".[dev]"
    - python quality_gates_validator.py
    - python -m pytest tests/ -v --cov=hypervector
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -f production_ready/docker/Dockerfile -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE
  only:
    - main

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl apply -f production_ready/kubernetes/
    - kubectl set image deployment/hypervector-lab hypervector-lab=$DOCKER_IMAGE -n $KUBE_NAMESPACE
    - kubectl rollout status deployment/hypervector-lab -n $KUBE_NAMESPACE
  environment:
    name: production
  only:
    - main
'''
        
        with open(cicd_dir / 'gitlab-ci.yml', 'w') as f:
            f.write(gitlab_ci)
        
        print("✓ Generated CI/CD pipeline configurations")
        return True
    
    def _generate_environment_configs(self) -> bool:
        """Generate environment-specific configurations."""
        configs_dir = self.deployment_dir / 'configs'
        
        # Development environment
        dev_config = {
            "environment": "development",
            "log_level": "DEBUG",
            "debug": True,
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "hypervector_dev"
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090
            },
            "security": {
                "cors_origins": ["http://localhost:3000"],
                "require_auth": False
            }
        }
        
        with open(configs_dir / 'development.json', 'w') as f:
            json.dump(dev_config, f, indent=2)
        
        # Production environment
        prod_config = {
            "environment": "production",
            "log_level": "INFO",
            "debug": False,
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
                "name": "${DB_NAME}",
                "user": "${DB_USER}",
                "password": "${DB_PASSWORD}"
            },
            "redis": {
                "host": "${REDIS_HOST}",
                "port": "${REDIS_PORT}",
                "password": "${REDIS_PASSWORD}"
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090,
                "prometheus_endpoint": "/metrics"
            },
            "security": {
                "cors_origins": ["${ALLOWED_ORIGINS}"],
                "require_auth": True,
                "jwt_secret": "${JWT_SECRET}",
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 100
                }
            },
            "scaling": {
                "auto_scale": True,
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 70
            }
        }
        
        with open(configs_dir / 'production.json', 'w') as f:
            json.dump(prod_config, f, indent=2)
        
        # Staging environment
        staging_config = {
            "environment": "staging",
            "log_level": "INFO",
            "debug": False,
            "database": {
                "host": "${STAGING_DB_HOST}",
                "port": 5432,
                "name": "hypervector_staging"
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090
            },
            "security": {
                "cors_origins": ["https://staging.hypervector.example.com"],
                "require_auth": True
            }
        }
        
        with open(configs_dir / 'staging.json', 'w') as f:
            json.dump(staging_config, f, indent=2)
        
        print("✓ Generated environment configurations")
        return True
    
    def _create_deployment_scripts(self) -> bool:
        """Create deployment automation scripts."""
        scripts_dir = self.deployment_dir / 'scripts'
        
        # Main deployment script
        deploy_script = '''#!/bin/bash
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
'''
        
        with open(scripts_dir / 'deploy.sh', 'w') as f:
            f.write(deploy_script)
        os.chmod(scripts_dir / 'deploy.sh', 0o755)
        
        # Health check script
        health_check_script = '''#!/bin/bash
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
'''
        
        with open(scripts_dir / 'health-check.sh', 'w') as f:
            f.write(health_check_script)
        os.chmod(scripts_dir / 'health-check.sh', 0o755)
        
        # Rollback script
        rollback_script = '''#!/bin/bash
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
'''
        
        with open(scripts_dir / 'rollback.sh', 'w') as f:
            f.write(rollback_script)
        os.chmod(scripts_dir / 'rollback.sh', 0o755)
        
        # Local development script
        local_dev_script = '''#!/bin/bash
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
'''
        
        with open(scripts_dir / 'start-dev.sh', 'w') as f:
            f.write(local_dev_script)
        os.chmod(scripts_dir / 'start-dev.sh', 0o755)
        
        print("✓ Generated deployment scripts")
        return True
    
    def _validate_deployment(self) -> bool:
        """Validate deployment readiness."""
        checks = [
            ("Docker files", lambda: self._check_files(['docker/Dockerfile', 'docker/docker-compose.yml'])),
            ("Kubernetes manifests", lambda: self._check_files(['kubernetes/deployment.yaml', 'kubernetes/service.yaml'])),
            ("Monitoring config", lambda: self._check_files(['monitoring/prometheus.yml'])),
            ("Deployment scripts", lambda: self._check_files(['scripts/deploy.sh'])),
            ("Environment configs", lambda: self._check_files(['configs/production.json']))
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if check_func():
                    print(f"✓ {check_name}")
                else:
                    print(f"✗ {check_name}")
                    all_passed = False
            except Exception as e:
                print(f"✗ {check_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def _check_files(self, file_paths: List[str]) -> bool:
        """Check if files exist in deployment directory."""
        for file_path in file_paths:
            full_path = self.deployment_dir / file_path
            if not full_path.exists():
                return False
        return True


def main():
    """Main execution function."""
    suite = ProductionDeploymentSuite()
    success = suite.prepare_complete_deployment()
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)