#!/bin/bash
set -e

# Start MinIO server with full path
/usr/local/bin/minio server --console-address ":9001" /data &

# Store the MinIO server PID
MINIO_PID=$!

# Wait for MinIO to be ready with timeout
echo "Waiting for MinIO to be ready..."
TIMEOUT=30
COUNTER=0
until curl -sf "http://localhost:9000/minio/health/live" > /dev/null 2>&1; do
    sleep 1
    COUNTER=$((COUNTER + 1))
    if [ $COUNTER -eq $TIMEOUT ]; then
        echo "Timeout waiting for MinIO to start"
        exit 1
    fi
    echo "Waiting... ($COUNTER/$TIMEOUT)"
done

echo "MinIO is ready!"

# Run the initialization script
echo "Initializing MinIO..."
python3 /usr/local/bin/init_minio.py

# Check if initialization was successful
if [ $? -ne 0 ]; then
    echo "MinIO initialization failed"
    exit 1
fi

echo "MinIO initialization completed successfully"

# Wait for the MinIO process
wait $MINIO_PID