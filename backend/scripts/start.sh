#!/bin/bash
set -e

echo "Starting Citation Network Explorer API..."

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

# Run database migrations if needed
echo "Running database setup..."
# python scripts/setup_db.py

# Start the application
echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4