#!/bin/bash

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! pg_isready -h postgres -p 5432 -U postgres -d forecasting_db -q; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 2
done

echo "PostgreSQL is ready!"

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -f http://ollama:11434/api/tags >/dev/null 2>&1; do
    echo "Ollama is unavailable - sleeping"
    sleep 5
done

echo "Ollama is ready!"

# Verify the model is available
echo "Checking if llama3.1 model is available..."
if ! curl -s http://ollama:11434/api/tags | grep -q "llama3.1"; then
    echo "Model llama3.1 not found, pulling it now..."
    curl -X POST http://ollama:11434/api/pull -d '{"name": "llama3.1"}'
    echo "Model pull initiated, waiting for completion..."
    sleep 30
fi

echo "Model is ready!"

# Setup the database tables
echo "Setting up the database..."
python setup_database.py

# Start the application
echo "Starting the application..."
uvicorn main:app --host 0.0.0.0 --port 8000