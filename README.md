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
├── streamlit_app.py
└── README.md
```

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- YOLO model weights (YOLOv8 recommended)

## Installation and Setup

### Setting up Python Virtual Environment

#### For Windows

1. **Install Python**
   - Download and install Python from [python.org](https://www.python.org/downloads/windows/)
   - Select "Add Python to PATH" during installation

2. **Create Virtual Environment**
   ```cmd
   # Open Command Prompt or PowerShell
   cd path\to\project
   python -m venv env
   ```

3. **Activate Virtual Environment**
   ```cmd
   # For Command Prompt
   env\Scripts\activate

   # For PowerShell
   .\env\Scripts\Activate.ps1
   ```

4. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

#### For macOS/Linux

1. **Install Python (if not already installed)**
   ```bash
   # For macOS with Homebrew
   brew install python

   # For Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Create Virtual Environment**
   ```bash
   cd path/to/project
   python3 -m venv env
   ```

3. **Activate Virtual Environment**
   ```bash
   source env/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Download YOLO Model

```bash
# After activating the virtual environment
python setup_model.py
```

### Running the Application

#### Run FastAPI Server
```bash
# Activate virtual environment first
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Run Streamlit Web Application
```bash
# Open another terminal and activate virtual environment
streamlit run streamlit_app.py
```

The Streamlit web application will automatically open at http://localhost:8501

## Quick Start

1. **Clone and setup the project**
   ```bash
   git clone <your-repo>
   cd yolo-fastapi-deployment
   ```

2. **Install dependencies (for local development)**
   ```bash
   # After creating and activating the virtual environment as described above
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

### Docker Deployment

#### Building and Running with Docker

1. **Build the Docker image**
   ```bash
   # Build the image with a tag
   docker build -t yolo-detection-api .
   
   # Check that the image was created
   docker images
   ```

2. **Run the Docker container**
   ```bash
   # Run the container, mapping port 8000
   docker run -p 8000:8000 --name yolo-api yolo-detection-api
   ```

3. **Stop the container**
   ```bash
   docker stop yolo-api
   ```

4. **Remove the container**
   ```bash
   docker rm yolo-api
   ```

#### Using Docker Compose

1. **Build and start services**
   ```bash
   docker-compose up --build
   ```

2. **Run in detached mode (background)**
   ```bash
   docker-compose up -d
   ```

3. **View logs**
   ```bash
   docker-compose logs -f
   ```

4. **Stop services**
   ```bash
   docker-compose down
   ```

#### Docker Compose Configuration

The `docker-compose.yml` file includes both the FastAPI service and a Streamlit web interface:

```yaml
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=models/yolov8n.pt
      - CONFIDENCE_THRESHOLD=0.5
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  web:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000/predict
    restart: unless-stopped
```

> Note: You may need to create a separate Dockerfile for the Streamlit application (Dockerfile.streamlit) if you want to run both services with Docker Compose.

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