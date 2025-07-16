import asyncio
from ultralytics import YOLO
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class YOLOModel:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
    
    async def load_model(self):
        """Load YOLO model asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, YOLO, self.model_path)
            self.is_loaded = True
            logger.info(f"YOLO model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    async def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run YOLO prediction on image
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Run prediction in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self._run_inference, 
                img_array
            )
            
            return self._parse_results(results)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e
    
    def _run_inference(self, img_array: np.ndarray):
        """Run YOLO inference"""
        return self.model(img_array, conf=self.confidence_threshold)
    
    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """Parse YOLO results into structured format"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        "class_id": int(box.cls.item()),
                        "class_name": result.names[int(box.cls.item())],
                        "confidence": float(box.conf.item()),
                        "bbox": {
                            "x1": float(box.xyxy[0][0].item()),
                            "y1": float(box.xyxy[0][1].item()),
                            "x2": float(box.xyxy[0][2].item()),
                            "y2": float(box.xyxy[0][3].item())
                        }
                    }
                    detections.append(detection)
        
        return detections