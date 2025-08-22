# 🚀 HyperVector-Lab Production Deployment Guide

**Version**: 1.0.0  
**Updated**: August 22, 2025  
**Target**: Production Enterprise Deployment

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Production Deployment](#kubernetes-production-deployment)
6. [Global Multi-Region Setup](#global-multi-region-setup)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security Configuration](#security-configuration)
9. [Compliance & Data Protection](#compliance--data-protection)
10. [Performance Tuning](#performance-tuning)
11. [Troubleshooting](#troubleshooting)
12. [Maintenance & Updates](#maintenance--updates)

---

## 🚀 Quick Start

### One-Command Production Deployment
```bash
# Clone repository
git clone https://github.com/danieleschmidt/hypervector-lab.git
cd hypervector-lab

# Deploy to Kubernetes (requires kubectl configured)
./scripts/deploy-production.sh --region us-west-2 --replicas 3

# Verify deployment
kubectl get pods -l app=hypervector-lab
```

### Verification Test
```bash
# Test basic functionality
python3 -c "
from hypervector import HDCSystem
hdc = HDCSystem(dim=10000)
print('✅ HyperVector-Lab deployment successful!')
"
```

---

## 📦 Prerequisites

### System Requirements
```yaml
Minimum Requirements:
  CPU: 4 cores, 2.4 GHz
  RAM: 8 GB
  Storage: 50 GB SSD
  Network: 1 Gbps

Recommended for Production:
  CPU: 16+ cores, 3.0+ GHz
  RAM: 32+ GB  
  Storage: 200+ GB NVMe SSD
  Network: 10+ Gbps
  GPU: Optional (NVIDIA T4/V100/A100)
```

### Software Dependencies
```bash
# Required Software
- Docker 20.10+
- Kubernetes 1.21+
- Python 3.9+
- PyTorch 2.0+

# Optional Enhancements
- NVIDIA Docker (for GPU acceleration)
- Prometheus & Grafana (monitoring)
- HashiCorp Vault (secrets management)
```

### Cloud Provider Support
- ✅ **AWS** (EKS, EC2, S3, RDS)
- ✅ **Google Cloud** (GKE, Compute Engine, Cloud Storage)
- ✅ **Azure** (AKS, Virtual Machines, Blob Storage)
- ✅ **On-Premises** (K8s, OpenShift, Docker Swarm)

---

## 💻 Local Development Setup

### 1. Environment Preparation
```bash
# Create virtual environment
python3 -m venv hdc_env
source hdc_env/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import hypervector; print(hypervector.get_version())"
```

### 2. Development Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
export HDC_ENVIRONMENT=development
export HDC_LOG_LEVEL=DEBUG
export HDC_DEVICE=cpu  # or 'cuda' for GPU
export HDC_DIMENSION=10000
```

### 3. Run Development Server
```bash
# Start development server
python -m hypervector.server --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v

# Performance benchmark
python -m hypervector.benchmark --quick
```

---

## 🐳 Docker Deployment

### 1. Build Production Image
```bash
# Build optimized production image
docker build -f Dockerfile.production -t hypervector-lab:latest .

# Multi-architecture build (ARM64 + AMD64)
docker buildx build --platform linux/amd64,linux/arm64 -t hypervector-lab:latest .
```

### 2. Docker Compose Deployment
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  hypervector-lab:
    image: hypervector-lab:latest
    ports:
      - "8000:8000"
    environment:
      - HDC_ENVIRONMENT=production
      - HDC_WORKERS=4
      - HDC_DIMENSION=10000
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_password
```

### 3. Start Services
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose ps
curl http://localhost:8000/health
```

---

## ☸️ Kubernetes Production Deployment

### 1. Namespace Setup
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hypervector-lab
  labels:
    name: hypervector-lab
    tier: production
```

### 2. ConfigMap Configuration
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hdc-config
  namespace: hypervector-lab
data:
  HDC_ENVIRONMENT: "production"
  HDC_DIMENSION: "10000"
  HDC_WORKERS: "8"
  HDC_BATCH_SIZE: "100"
  HDC_LOG_LEVEL: "INFO"
```

### 3. Production Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypervector-lab
  namespace: hypervector-lab
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
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
            name: hdc-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### 4. Service & Ingress
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: hypervector-lab-service
  namespace: hypervector-lab
spec:
  selector:
    app: hypervector-lab
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hypervector-lab-ingress
  namespace: hypervector-lab
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - hdc.yourdomain.com
    secretName: hdc-tls
  rules:
  - host: hdc.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hypervector-lab-service
            port:
              number: 80
```

### 5. Horizontal Pod Autoscaler
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hypervector-lab-hpa
  namespace: hypervector-lab
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hypervector-lab
  minReplicas: 3
  maxReplicas: 100
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
```

### 6. Deploy to Kubernetes
```bash
# Apply all configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl get all -n hypervector-lab
kubectl logs -f deployment/hypervector-lab -n hypervector-lab
```

---

## 🌍 Global Multi-Region Setup

### 1. Region Configuration
```yaml
regions:
  us-west-2:
    provider: aws
    clusters: 
      - production-usw2-1
      - production-usw2-2
    compliance: [CCPA]
    
  eu-west-1:
    provider: aws
    clusters:
      - production-euw1-1
      - production-euw1-2
    compliance: [GDPR]
    
  ap-southeast-1:
    provider: aws
    clusters:
      - production-apse1-1
    compliance: [PDPA]
```

### 2. Multi-Region Deployment Script
```bash
#!/bin/bash
# deploy-global.sh

REGIONS=("us-west-2" "eu-west-1" "ap-southeast-1")

for region in "${REGIONS[@]}"; do
    echo "Deploying to $region..."
    
    # Switch kubectl context
    kubectl config use-context $region
    
    # Deploy with region-specific config
    envsubst < k8s/deployment.yaml | kubectl apply -f -
    
    # Verify deployment
    kubectl rollout status deployment/hypervector-lab -n hypervector-lab
    
    echo "✅ $region deployment complete"
done

echo "🌍 Global deployment successful!"
```

### 3. Global Load Balancer
```yaml
# global-lb.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: hypervector-global
spec:
  hosts:
  - hdc-global.yourdomain.com
  http:
  - match:
    - headers:
        x-region:
          exact: us
    route:
    - destination:
        host: hdc-us.yourdomain.com
  - match:
    - headers:
        x-region:
          exact: eu
    route:
    - destination:
        host: hdc-eu.yourdomain.com
  - route:
    - destination:
        host: hdc-us.yourdomain.com
      weight: 50
    - destination:
        host: hdc-eu.yourdomain.com
      weight: 30
    - destination:
        host: hdc-ap.yourdomain.com
      weight: 20
```

---

## 📊 Monitoring & Observability

### 1. Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'hypervector-lab'
  static_configs:
  - targets: ['hypervector-lab-service:80']
  metrics_path: /metrics
  scrape_interval: 5s

- job_name: 'kubernetes-pods'
  kubernetes_sd_configs:
  - role: pod
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
    action: keep
    regex: true
```

### 2. Grafana Dashboard
```json
{
  "dashboard": {
    "title": "HyperVector-Lab Performance",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(hdc_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, hdc_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(hdc_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3. Alerting Rules
```yaml
# alerts.yml
groups:
- name: hypervector-lab
  rules:
  - alert: HighErrorRate
    expr: rate(hdc_errors_total[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, hdc_request_duration_seconds_bucket) > 0.1
    for: 5m
    annotations:
      summary: "High latency detected"
      
  - alert: PodCrashLooping
    expr: rate(kube_pod_container_status_restarts_total[5m]) > 0
    for: 1m
    annotations:
      summary: "Pod is crash looping"
```

---

## 🔐 Security Configuration

### 1. Network Policies
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hypervector-lab-netpol
  namespace: hypervector-lab
spec:
  podSelector:
    matchLabels:
      app: hypervector-lab
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### 2. Pod Security Policy
```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: hypervector-lab-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 3. Secrets Management
```bash
# Create secrets
kubectl create secret generic hdc-secrets \
  --from-literal=database-url="postgresql://user:pass@db:5432/hdc" \
  --from-literal=redis-url="redis://redis:6379/0" \
  --from-literal=encryption-key="your-encryption-key" \
  --namespace hypervector-lab

# Reference in deployment
env:
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: hdc-secrets
      key: database-url
```

---

## ⚖️ Compliance & Data Protection

### 1. GDPR Configuration
```yaml
# gdpr-config.yaml
compliance:
  gdpr:
    enabled: true
    data_retention_days: 2555  # 7 years
    consent_required: true
    right_to_erasure: true
    data_portability: true
    lawful_basis: "legitimate_interests"
    dpo_contact: "dpo@yourdomain.com"
```

### 2. Data Anonymization Pipeline
```python
# Auto-anonymization for EU traffic
from hypervector.utils.global_compliance import get_compliance_framework

compliance = get_compliance_framework()

def process_eu_data(data, user_id):
    return compliance.process_data_with_compliance(
        data=data,
        region="EU",
        subject_id=user_id,
        data_classification=DataClassification.PERSONAL,
        processing_purpose=ProcessingPurpose.RESEARCH
    )
```

### 3. Audit Logging
```yaml
# audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: RequestResponse
  namespaces: ["hypervector-lab"]
  resources:
  - group: ""
    resources: ["configmaps", "secrets"]
  - group: "apps"
    resources: ["deployments"]
```

---

## ⚡ Performance Tuning

### 1. Resource Optimization
```yaml
# Optimized resource allocation
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
    
# JVM tuning for PyTorch
env:
- name: MALLOC_CONF
  value: "background_thread:true,metadata_thp:auto"
- name: OMP_NUM_THREADS
  value: "4"
```

### 2. Caching Strategy
```python
# Redis configuration for caching
REDIS_CONFIG = {
    'host': 'redis-cluster',
    'port': 6379,
    'db': 0,
    'max_connections': 100,
    'socket_timeout': 5,
    'retry_on_timeout': True
}

# Enable operation caching
HDC_CACHE_CONFIG = {
    'enable_cache': True,
    'cache_size': 10000,
    'cache_ttl': 3600,  # 1 hour
    'cache_backend': 'redis'
}
```

### 3. Database Optimization
```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET max_connections = 200;

-- Indexes for performance
CREATE INDEX CONCURRENTLY idx_hypervector_metadata 
ON hypervectors USING GIN (metadata);

CREATE INDEX CONCURRENTLY idx_operations_timestamp 
ON operations (created_at DESC);
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Memory Issues
```bash
# Symptoms: OOM kills, high memory usage
# Solutions:
kubectl top pods -n hypervector-lab  # Check memory usage
kubectl describe pod <pod-name> -n hypervector-lab  # Check limits

# Increase memory limits
resources:
  limits:
    memory: "16Gi"  # Increase from 8Gi
```

#### 2. Performance Degradation
```bash
# Check CPU throttling
kubectl top nodes
kubectl describe node <node-name>

# Scale horizontally
kubectl scale deployment hypervector-lab --replicas=10 -n hypervector-lab

# Check HPA status
kubectl get hpa -n hypervector-lab
```

#### 3. Connection Issues
```bash
# Check service connectivity
kubectl get svc -n hypervector-lab
kubectl get endpoints -n hypervector-lab

# Port forward for debugging
kubectl port-forward svc/hypervector-lab-service 8000:80 -n hypervector-lab

# Test connectivity
curl http://localhost:8000/health
```

#### 4. Configuration Problems
```bash
# Check configmap
kubectl get configmap hdc-config -o yaml -n hypervector-lab

# Check secrets
kubectl get secrets -n hypervector-lab

# Restart deployment
kubectl rollout restart deployment/hypervector-lab -n hypervector-lab
```

### Debug Commands
```bash
# Get pod logs
kubectl logs -f deployment/hypervector-lab -n hypervector-lab

# Execute into pod
kubectl exec -it <pod-name> -n hypervector-lab -- /bin/bash

# Check resource usage
kubectl top pods -n hypervector-lab

# Get detailed pod info
kubectl describe pod <pod-name> -n hypervector-lab
```

---

## 🔄 Maintenance & Updates

### 1. Rolling Updates
```bash
# Update image
kubectl set image deployment/hypervector-lab \
  hypervector-lab=hypervector-lab:v1.1.0 \
  -n hypervector-lab

# Monitor rollout
kubectl rollout status deployment/hypervector-lab -n hypervector-lab

# Rollback if needed
kubectl rollout undo deployment/hypervector-lab -n hypervector-lab
```

### 2. Backup Strategy
```bash
# Backup persistent data
kubectl exec -n hypervector-lab <pod-name> -- \
  tar czf - /app/data | \
  gsutil cp - gs://backup-bucket/hdc-backup-$(date +%Y%m%d).tar.gz

# Backup configurations
kubectl get all -n hypervector-lab -o yaml > hdc-backup.yaml
```

### 3. Health Checks
```bash
# Automated health check script
#!/bin/bash
# health-check.sh

NAMESPACE="hypervector-lab"
SERVICE_URL="http://hdc.yourdomain.com"

# Check pod health
UNHEALTHY_PODS=$(kubectl get pods -n $NAMESPACE | grep -v Running | wc -l)
if [ $UNHEALTHY_PODS -gt 1 ]; then
    echo "⚠️  Warning: $UNHEALTHY_PODS unhealthy pods"
fi

# Check service health
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $SERVICE_URL/health)
if [ $HTTP_STATUS -ne 200 ]; then
    echo "❌ Service health check failed: $HTTP_STATUS"
    exit 1
fi

echo "✅ All health checks passed"
```

---

## 📚 Additional Resources

### Documentation Links
- **API Documentation**: `/docs` endpoint
- **Monitoring Dashboards**: Grafana at `:3000`
- **Metrics Endpoint**: `/metrics` for Prometheus
- **Health Checks**: `/health` and `/ready`

### Support Contacts
- **Technical Support**: tech-support@yourdomain.com
- **Security Issues**: security@yourdomain.com  
- **Data Protection**: dpo@yourdomain.com

### Useful Commands
```bash
# Quick status check
kubectl get all -n hypervector-lab

# Scale deployment
kubectl scale deployment hypervector-lab --replicas=5 -n hypervector-lab

# View logs
kubectl logs -f -l app=hypervector-lab -n hypervector-lab

# Port forward for debugging
kubectl port-forward svc/hypervector-lab-service 8000:80 -n hypervector-lab
```

---

**🎉 Congratulations! You now have a production-ready HyperVector-Lab deployment.**

For additional support, please refer to our [GitHub repository](https://github.com/danieleschmidt/hypervector-lab) or contact our support team.

---

*Generated by Terragon Labs Production Engineering Team*  
*Last Updated: August 22, 2025*