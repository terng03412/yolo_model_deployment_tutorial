from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import os
from typing import List, Dict, Any

from app.models.yolo_model import YOLOModel
from app.utils.image_processing import process_image, validate_image

app = FastAPI(
    title="YOLO Object Detection API",
    description="FastAPI service for YOLO object detection",
    version="1.0.0"
)

# Initialize YOLO model
model_path = os.getenv("MODEL_PATH", "models/yolov8n.pt")
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
yolo_model = YOLOModel(model_path, confidence_threshold)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    await yolo_model.load_model()

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "YOLO FastAPI service is running"}

@app.post("/predict")
async def predict_objects(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict objects in uploaded image using YOLO model
    """
    try:
        # Validate file
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = process_image(image)
        
        # Run prediction
        results = await yolo_model.predict(processed_image)
        
        return {
            "filename": file.filename,
            "predictions": results,
            "total_objects": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "model_path": model_path,
        "confidence_threshold": confidence_threshold,
        "model_loaded": yolo_model.is_loaded
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )