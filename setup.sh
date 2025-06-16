#!/bin/bash

# Real Estate AI - Complete Setup Script

echo "🏠 Real Estate AI - Docker Setup"
echo "================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Create required directories
echo "📁 Creating required directories..."
mkdir -p dags logs plugins data/models plots config

# Create placeholder files to ensure directories are tracked
touch data/.gitkeep
touch plots/.gitkeep

# Set proper permissions (Linux/Mac)
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🔒 Setting permissions..."
    chmod +x start.sh
    chmod +x init-scripts/init-db.sh
    
    # Set AIRFLOW_UID in .env
    echo "AIRFLOW_UID=$(id -u)" >> .env
fi

# Check if data file exists
if [ ! -f "data/real_estate_data.csv" ]; then
    echo "⚠️  Warning: data/real_estate_data.csv not found"
    echo "   Please add your real estate data file to the data/ directory"
fi

echo "🐳 Building Docker images..."
docker-compose build

echo "🗄️  Initializing Airflow database..."
docker-compose up airflow-init

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

echo "📊 Checking service status..."
docker-compose ps

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🌐 Access Points:"
echo "  • Airflow Web UI: http://localhost:8080"
echo "    Username: admin | Password: admin"
echo ""
echo "  • PostgreSQL: localhost:5432"
echo "    Airflow DB: airflow/airflow"
echo "    Real Estate DB: real_estate_user/real_estate_pass@real_estate_db"
echo ""
echo "🔧 Management Commands:"
echo "  • View logs: docker-compose logs -f"
echo "  • Stop services: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Monitor Celery: docker-compose --profile flower up -d"
echo ""
echo "📋 Next Steps:"
echo "  1. Upload your data to data/real_estate_data.csv"
echo "  2. Access Airflow UI and enable the DAGs"
echo "  3. Monitor pipeline execution"
echo ""
