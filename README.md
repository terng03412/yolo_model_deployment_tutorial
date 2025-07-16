# YOLO Model Deployment with FastAPI and Docker

This guide provides step-by-step instructions for deploying a YOLO object detection model using FastAPI, Docker, and uvicorn.

## Project Structure

```
yolo-fastapi-deployment/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── yolo_model.py
│   └── utils/
│       ├── __init__.py
│       └── image_processing.py
├── models/
│   └── yolov8n.pt  # Your YOLO model weights
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- YOLO model weights (YOLOv8 recommended)

## Quick Start

1. **Clone and setup the project**
   ```bash
   git clone <your-repo>
   cd yolo-fastapi-deployment
   ```

2. **Install dependencies (for local development)**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your YOLO model**
   - Place your YOLO model weights in the `models/` directory
   - Update the model path in `app/models/yolo_model.py`

4. **Run locally with uvicorn**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Run with Docker**
   ```bash
   docker-compose up --build
   ```

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Object detection on uploaded image
- `GET /docs` - Interactive API documentation (Swagger UI)

## Usage Examples

### Using curl
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

### Using Python requests
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Configuration

### Environment Variables
- `MODEL_PATH`: Path to YOLO model weights (default: "models/yolov8n.pt")
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.5)
- `MAX_IMAGE_SIZE`: Maximum image size in MB (default: 10)

## Deployment Options

### Local Development
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production with Docker
```bash
docker-compose up -d
```

### Production with uvicorn (without Docker)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Performance Optimization

- Use GPU acceleration by installing `torch` with CUDA support
- Adjust worker count based on your server specifications
- Implement caching for frequently processed images
- Use async/await for I/O operations

## Troubleshooting

### Common Issues
1. **Model loading errors**: Ensure model path is correct and weights are compatible
2. **Memory issues**: Reduce image size or adjust batch processing
3. **Port conflicts**: Change port in docker-compose.yml or uvicorn command

### Logs
```bash
# Docker logs
docker-compose logs -f

# Local development
# Logs will appear in terminal with --reload flag
```

## Security Considerations

- Implement rate limiting
- Validate file types and sizes
- Use HTTPS in production
- Consider authentication for sensitive deployments

## Next Steps

- Add batch processing capabilities
- Implement result caching
- Add monitoring and logging
- Set up CI/CD pipeline