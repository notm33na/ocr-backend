# OCR Backend API

A FastAPI-based OCR (Optical Character Recognition) backend service that supports both English and Urdu text extraction from images using custom TrOCR-LoRA and UTRNet models.

## Features

- **English OCR**: Uses custom TrOCR-LoRA model (`trocr-lora-unified`) with paragraph processing and medical text corrections
- **Urdu OCR**: Uses UTRNet high-resolution text recognition + YOLOv8 text detection
- **RESTful API**: Clean FastAPI endpoints with automatic documentation
- **CORS Support**: Configured for cross-origin requests from mobile/web apps
- **Health Checks**: Built-in health monitoring endpoints
- **Error Handling**: Comprehensive error handling with proper HTTP status codes
- **Visualization**: Optional visualization output for English OCR (shows detected text regions)

## Model Information

### English OCR (TrOCR-LoRA)
- **Model Checkpoint**: `trocr-lora-unified`
- **Base Model**: `microsoft/trocr-base-handwritten`
- **Features**:
  - Paragraph processing with line-by-line detection
  - YOLOv8 text region detection
  - Medical term corrections (optional)
  - Visualization of detected regions (optional)

### Urdu OCR (UTRNet + YOLO)
- **Detection Model**: `yolov8m_UrduDoc (1).pt` (YOLOv8 medium, fine-tuned on UrduDoc)
- **Recognition Model**: `saved_models/UTRNet-Large/best_norm_ED.pth`
- **Image Dimensions**: 32x400 (height x width)
- **Confidence Threshold**: 0.2
- **Architecture**: HRNet backbone + DBiLSTM + CTC prediction

## API Endpoints

### Health Check Endpoints

#### `GET /`
Returns API status and available endpoints.

**Response:**
```json
{
  "status": "healthy",
  "message": "OCR Backend API",
  "endpoints": {
    "health": "/health",
    "ocr": "/ocr",
    "docs": "/docs"
  }
}
```

#### `GET /health`
Detailed health check endpoint for platform monitoring.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "english": true,
    "urdu": true
  },
  "ready": true
}
```

### OCR Endpoint

#### `POST /ocr`
Perform OCR on a base64-encoded image.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "language": "english",  // or "urdu", defaults to "english"
  "medical_corrections": false,  // optional, only for English, default: false
  "visualize": false  // optional, only for English, returns visualization image, default: false
}
```

**Response:**
```json
{
  "text": "Extracted text from image",
  "visualization": "base64_encoded_visualization_image"  // optional, only if visualize=true
}
```

**Example using curl:**
```bash
curl -X POST "https://your-app.railway.app/ocr" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "language": "english",
    "medical_corrections": true,
    "visualize": true
  }'
```

**Error Responses:**

- `400 Bad Request`: Invalid image data or unsupported language
- `500 Internal Server Error`: OCR processing failed
- `503 Service Unavailable`: Models not loaded yet

## Project Structure

```
ocr-backend/
├── main.py                           # FastAPI application
├── requirements.txt                  # Python dependencies
├── Procfile                         # Railway/Render start command
├── runtime.txt                      # Python version specification
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
├── trocr-lora-unified/              # TrOCR-LoRA model checkpoint (if stored locally)
└── (models are in parent directories)

Parent directory structure:
├── trocr-base-printed/
│   ├── paragraph_ocr.py             # English OCR script
│   ├── inference.py                 # TrOCR model loading
│   └── trocr-lora-unified/          # Model checkpoint
├── urdu-text-detection/
│   └── yolov8m_UrduDoc (1).pt      # YOLO detection model
└── UTRNet-High-Resolution-Urdu-Text-Recognition/
    ├── end_to_end_ocr.py            # Urdu OCR script
    ├── saved_models/
    │   └── UTRNet-Large/
    │       └── best_norm_ED.pth     # UTRNet recognition model
    └── UrduGlyphs.txt               # Character vocabulary
```

## Local Development

### Prerequisites

- Python 3.11.0
- pip or conda
- Model files in correct locations (see Project Structure above)

### Installation

1. Navigate to the `ocr-backend` directory:
```bash
cd ocr-backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:

**For CPU-only (cloud deployment):**
```bash
# Install PyTorch CPU-only first
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.17.0

# Then install other dependencies
pip install -r requirements.txt
```

**Note:** If you encounter Rust compilation errors with `tokenizers`, the package will be installed automatically by `transformers` with a compatible version that has pre-built wheels. If issues persist, you can install Rust manually from [rustup.rs](https://rustup.rs/) or use a pre-built wheel:
```bash
pip install tokenizers --only-binary :all:
```

**Or use the installation script:**
```bash
# Windows PowerShell
.\install.ps1

# Linux/Mac
chmod +x install.sh
./install.sh
```

**Note:** For local development with GPU support, install PyTorch with CUDA:
```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
pip install -r requirements.txt
```

### Running Locally

1. Set the PORT environment variable (optional, defaults to 8000):
```bash
# Windows PowerShell
$env:PORT=8000

# Linux/Mac
export PORT=8000
```

2. Start the server:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --reload
```

Or using Python:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Access the API:
- API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Testing the API

#### Using the test script (Recommended):
```bash
# Python test script (works on all platforms)
python test_api.py ../61.jpg english
python test_api.py ../61.jpg english true  # with medical corrections
python test_api.py ../61.jpg english true true  # with medical corrections and visualization
python test_api.py ../urdu-text-detection/test.jpg urdu
```

**PowerShell (Windows):**
```powershell
.\test_api.ps1 ..\61.jpg english
.\test_api.ps1 ..\61.jpg english -MedicalCorrections
.\test_api.ps1 ..\61.jpg english -MedicalCorrections -Visualize
.\test_api.ps1 ..\urdu-text-detection\test.jpg urdu
```

#### Using Python directly:
```python
import requests
import base64

# Read and encode image
with open("test_image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make OCR request
response = requests.post(
    "http://localhost:8000/ocr",
    json={
        "image": image_base64,
        "language": "english",
        "medical_corrections": True,
        "visualize": False
    }
)
print(response.json())
```

#### Using curl (Linux/Mac):
```bash
# Health check
curl http://localhost:8000/health

# OCR request (replace with actual base64 image)
curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "your_base64_image",
    "language": "english",
    "medical_corrections": true
  }'
```

#### Using PowerShell Invoke-RestMethod:
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

# OCR request (need to encode image to base64 first)
$imageBytes = [System.IO.File]::ReadAllBytes("path\to\image.jpg")
$imageBase64 = [System.Convert]::ToBase64String($imageBytes)
$body = @{image=$imageBase64; language="english"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/ocr" -Method Post -Body $body -ContentType "application/json"
```

## Cloud Deployment

### Model File Management

**Important:** Model files are large (500MB-2GB total) and may cause deployment issues. Consider these options:

1. **Git LFS (Recommended for small teams):**
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "trocr-lora-unified/**"
git lfs track "saved_models/**"
git add .gitattributes
```

2. **External Storage (Recommended for production):**
   - Upload models to cloud storage (S3, Google Cloud Storage, etc.)
   - Download models on server startup
   - See "Model Download on Startup" section below

3. **Model Hosting Services:**
   - Use HuggingFace Model Hub
   - Use cloud ML model serving platforms

### Railway Deployment

1. **Create a Railway account** at [railway.app](https://railway.app)

2. **Create a new project** and connect your Git repository

3. **Railway will automatically detect:**
   - Python from `runtime.txt`
   - Dependencies from `requirements.txt`
   - Start command from `Procfile`

4. **Environment Variables:**
   - `PORT` is automatically set by Railway
   - Optional: `MODEL_CACHE_DIR` for model storage
   - Optional: `MAX_IMAGE_SIZE` to limit image size

5. **Deploy:**
   - Push your code to the connected repository
   - Railway will automatically build and deploy
   - First deployment may take 15-20 minutes (model downloads/loading)

6. **Access your API:**
   - Railway provides a URL like: `https://your-app.railway.app`
   - Health check: `https://your-app.railway.app/health`
   - OCR endpoint: `https://your-app.railway.app/ocr`

### Render Deployment

1. **Create a Render account** at [render.com](https://render.com)

2. **Create a new Web Service:**
   - Connect your Git repository
   - Select the `ocr-backend` directory as the root

3. **Configure the service:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3
   - **Python Version:** 3.11.0 (from `runtime.txt`)

4. **Environment Variables:**
   - `PORT` is automatically set by Render
   - Optional: `MODEL_CACHE_DIR` for model storage

5. **Deploy:**
   - Render will build and deploy automatically
   - First deployment may take 15-20 minutes (model downloads/loading)

6. **Note:** Render's free tier has cold starts - the app may sleep after inactivity

### Model Download on Startup (Alternative Approach)

If models are stored externally, modify `main.py` to download them on startup:

```python
import os
import requests
from pathlib import Path

MODEL_URLS = {
    "trocr": os.getenv("TROCR_MODEL_URL", ""),
    "yolo": os.getenv("YOLO_MODEL_URL", ""),
    "utrnet": os.getenv("UTRNet_MODEL_URL", "")
}

def download_model(url: str, dest_path: Path):
    """Download model from URL if not exists"""
    if dest_path.exists():
        logger.info(f"Model already exists: {dest_path}")
        return
    
    logger.info(f"Downloading model from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Model downloaded: {dest_path}")

@app.on_event("startup")
async def load_models():
    # Download models if URLs are provided
    if MODEL_URLS["trocr"]:
        download_model(MODEL_URLS["trocr"], TROCR_CHECKPOINT)
    # ... similar for other models
```

### Important Notes for Cloud Deployment

- **Model Download:** First deployment will download/load models (~500MB-2GB), which may take 15-20 minutes
- **CPU-Only PyTorch:** The `requirements.txt` uses CPU-only PyTorch to reduce deployment size and cost
- **Memory Requirements:** Ensure your cloud plan has at least 2-4GB RAM for model loading
- **Startup Time:** Initial model loading takes 1-3 minutes on first request after deployment
- **Storage Limits:** Free tiers may have storage limits - consider using external storage for models
- **Git LFS Limits:** Free Git LFS has bandwidth limits - may not be suitable for frequent deployments

## API Documentation

Once the server is running, interactive API documentation is available at:
- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`

## Troubleshooting

### Models not loading
- Check logs for download/loading errors
- Ensure sufficient disk space (models are ~500MB-2GB)
- Verify model file paths are correct
- Check that model files exist in expected locations
- Verify internet connection during first deployment (if downloading)

### CORS errors
- The API is configured to allow all origins (`*`)
- For production, update `main.py` to restrict origins:
```python
allow_origins=["https://your-frontend-domain.com"]
```

### Port binding errors
- Ensure `PORT` environment variable is set
- Check that the port is not already in use

### Memory issues
- Cloud platforms may require at least 2-4GB RAM
- Consider upgrading your plan if models fail to load
- Use model quantization if available

### Model file not found errors
- Verify model paths in `main.py` match your directory structure
- Check that model files are committed to Git (or use Git LFS)
- For external storage, verify download URLs are correct

### Import errors
- Ensure parent directories (`trocr-base-printed`, `UTRNet-High-Resolution-Urdu-Text-Recognition`) are accessible
- Check that all required Python packages are installed
- Verify Python path includes necessary directories

## License

This project uses pre-trained models:
- **TrOCR**: Microsoft (Apache 2.0)
- **UTRNet**: Licensed under Creative Commons Attribution-NonCommercial 4.0 International License
- **YOLOv8**: AGPL-3.0 (Ultralytics)

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify model files are in correct locations
3. Test locally before deploying to cloud
4. Create an issue in the repository with logs and error details
