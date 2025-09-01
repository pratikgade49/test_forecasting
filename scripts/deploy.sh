#!/bin/bash

# Production deployment script
set -e

echo "🚀 Starting production deployment..."

# Check if required environment variables are set
if [ -z "$SECRET_KEY" ]; then
    echo "⚠️  WARNING: SECRET_KEY not set. Using default (not secure for production)"
fi

if [ -z "$DB_PASSWORD" ]; then
    echo "⚠️  WARNING: DB_PASSWORD not set. Using default (not secure for production)"
fi

# Pull latest images
echo "📦 Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose -f docker-compose.prod.yml down

# Start services
echo "🔄 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Application is healthy!"
else
    echo "❌ Application health check failed"
    docker-compose -f docker-compose.prod.yml logs app
    exit 1
fi

# Check AI service
if curl -f http://localhost:8000/ai/health >/dev/null 2>&1; then
    echo "✅ AI chat service is healthy!"
else
    echo "⚠️  AI chat service may not be ready yet (this is normal on first startup)"
fi

echo "🎉 Deployment completed successfully!"
echo "📱 Access your application at: http://localhost:8000"
echo "🤖 AI Chat is available in the bottom-right corner"
echo ""
echo "Default login: username=admin, password=admin123"
echo "Please change the default password after first login!"