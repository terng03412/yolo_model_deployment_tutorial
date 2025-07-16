#!/usr/bin/env python3
"""
Setup script to download YOLO model weights
"""

import os
from pathlib import Path
from ultralytics import YOLO

def setup_yolo_model():
    """Download and setup YOLO model"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "yolov8n.pt"
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return
    
    print("Downloading YOLOv8n model...")
    try:
        # This will download the model if it doesn't exist
        model = YOLO('yolov8n.pt')
        
        # Save to models directory
        print(f"Saving model to {model_path}")
        model.save(str(model_path))
        
        print("Model setup completed successfully!")
        
    except Exception as e:
        print(f"Error setting up model: {e}")
        raise

if __name__ == "__main__":
    setup_yolo_model()