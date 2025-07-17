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
        # Read image bytes first
        image_bytes = await file.read()
        
        # Reset file pointer for validation
        await file.seek(0)
        
        # Validate file
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check if we have image data
        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image data received")
        
        try:
            # Create BytesIO object and try to open image
            image_buffer = io.BytesIO(image_bytes)
            image = Image.open(image_buffer)
            
            # Verify the image by trying to load it
            image.verify()
            
            # Reopen the image for processing (verify() closes the image)
            image_buffer.seek(0)
            image = Image.open(image_buffer)
            
            # Store original image dimensions
            original_width, original_height = image.size
            
            # Process the image
            processed_image = process_image(image)
            
            # Store processed image dimensions
            processed_width, processed_height = processed_image.size
            
            # Calculate scale factors for bounding box correction
            scale_x = original_width / processed_width
            scale_y = original_height / processed_height
            
            # Run prediction
            results = await yolo_model.predict(processed_image)
            
            # Adjust bounding box coordinates to match original image
            for pred in results:
                pred["bbox"]["x1"] *= scale_x
                pred["bbox"]["y1"] *= scale_y
                pred["bbox"]["x2"] *= scale_x
                pred["bbox"]["y2"] *= scale_y
            
            return {
                "filename": file.filename,
                "predictions": results,
                "total_objects": len(results)
            }
            
        except Exception as img_error:
            # More specific error for image processing issues
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to process image: {str(img_error)}. Please ensure the file is a valid image format (JPEG, PNG, etc.)"
            )
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        # General error handling
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