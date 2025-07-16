#!/usr/bin/env python3
"""
Test script for YOLO FastAPI deployment
"""

import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"  # Replace with your test image

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_prediction(image_path: str):
    """Test prediction endpoint"""
    print(f"\nTesting prediction with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Test image {image_path} not found. Please provide a valid image path.")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Filename: {result['filename']}")
        print(f"Total objects detected: {result['total_objects']}")
        print("Predictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  {i+1}. {pred['class_name']} (confidence: {pred['confidence']:.2f})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("YOLO FastAPI Test Suite")
    print("=" * 30)
    
    # Test health check
    if not test_health_check():
        print("Health check failed. Is the server running?")
        return
    
    # Test model info
    test_model_info()
    
    # Test prediction
    test_prediction(TEST_IMAGE_PATH)
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()