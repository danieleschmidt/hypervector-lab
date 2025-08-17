#!/bin/bash
set -euo pipefail

# HyperVector-Lab Rollback Script

echo "🔄 Starting rollback procedure..."

DEPLOYMENT_TYPE=${1:-docker}
BACKUP_VERSION=${2:-latest}

echo "Rolling back $DEPLOYMENT_TYPE deployment to $BACKUP_VERSION"

case $DEPLOYMENT_TYPE in
    "docker")
        echo "🐳 Rolling back Docker deployment..."
        docker-compose down
        docker tag hypervector-lab:$BACKUP_VERSION hypervector-lab:latest
        docker-compose up -d
        ;;
    "kubernetes")
        echo "☸️ Rolling back Kubernetes deployment..."
        kubectl rollout undo deployment/hypervector-app -n hypervector
        ;;
    *)
        echo "❌ Unknown deployment type: $DEPLOYMENT_TYPE"
        exit 1
        ;;
esac

echo "✅ Rollback completed!"
