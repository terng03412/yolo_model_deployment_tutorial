#!/bin/bash

# YOLO FastAPI Deployment Startup Script

echo "Starting YOLO FastAPI deployment..."

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Download YOLOv8 model if not exists
if [ ! -f "models/yolov8n.pt" ]; then
    echo "Downloading YOLOv8n model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('models/yolov8n.pt')"
fi

# Start the application
echo "Starting uvicorn server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload