#!/bin/bash
set -e

echo "🚀 Starting HyperVector deployment..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t hypervector-app:latest .

# Deploy with docker-compose
echo "🐳 Deploying with Docker Compose..."
docker-compose up -d

echo "✅ Deployment completed!"
echo "🌐 Application should be available on the configured ports"
