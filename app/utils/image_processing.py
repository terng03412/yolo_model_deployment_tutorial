from PIL import Image
from fastapi import UploadFile
import os
from typing import Optional

# Configuration
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_DIMENSION = 1920

def validate_image(file: UploadFile) -> bool:
    """
    Validate uploaded image file
    """
    # Check file extension
    if not file.filename:
        return False
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check content type
    if not file.content_type or not file.content_type.startswith("image/"):
        return False
    
    return True

def process_image(image: Image.Image) -> Image.Image:
    """
    Process image for YOLO inference
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize if too large
    width, height = image.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = MAX_DIMENSION
            new_height = int((height * MAX_DIMENSION) / width)
        else:
            new_height = MAX_DIMENSION
            new_width = int((width * MAX_DIMENSION) / height)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def get_image_info(image: Image.Image) -> dict:
    """
    Get image information
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format
    }