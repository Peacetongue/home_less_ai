#!/bin/bash

# Real Estate AI - Docker Startup Script

echo "Starting Real Estate AI Docker Environment..."

# Create necessary directories
mkdir -p dags logs plugins data/models

# Set permissions for Airflow (Linux/WSL)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Setting permissions for Linux environment..."
    echo -e "AIRFLOW_UID=$(id -u)" > .env
    sudo chown -R $(id -u):$(id -g) dags logs plugins data
fi

# Initialize Airflow database
echo "Initializing Airflow..."
docker-compose up airflow-init

# Start the services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check service status
echo "Checking service status..."
docker-compose ps

echo ""
echo "=================================================="
echo "Real Estate AI Environment is starting up!"
echo "=================================================="
echo ""
echo "Services:"
echo "- Airflow Web UI: http://localhost:8080"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "- PostgreSQL Database: localhost:5432"
echo "  Username: airflow"
echo "  Password: airflow"
echo "  Database: airflow"
echo ""
echo "- Real Estate Database: localhost:5432"
echo "  Username: real_estate_user"
echo "  Password: real_estate_pass"
echo "  Database: real_estate_db"
echo ""
echo "- Flower (Celery monitoring): http://localhost:5555"
echo "  (Run: docker-compose --profile flower up -d)"
echo ""
echo "To view logs: docker-compose logs -f [service_name]"
echo "To stop: docker-compose down"
echo "To rebuild: docker-compose build --no-cache"
echo ""
