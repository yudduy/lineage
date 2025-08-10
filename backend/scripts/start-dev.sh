#!/bin/bash
set -e

echo "Starting Citation Network Explorer API (Development Mode)..."

# Wait for database to be ready
echo "Waiting for Neo4j to be ready..."
while ! nc -z neo4j 7687; do
    sleep 1
done
echo "Neo4j is ready!"

echo "Waiting for Redis to be ready..."
while ! nc -z redis 6379; do
    sleep 1
done
echo "Redis is ready!"

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Run database setup
echo "Setting up development database..."
# python scripts/setup_dev_db.py

# Start the application with hot reload
echo "Starting FastAPI application in development mode..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug