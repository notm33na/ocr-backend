"""
FastAPI OCR Backend
Integrates TrOCR-LoRA (English) and UTRNet + YOLO (Urdu) OCR systems
"""

import os
import sys
import base64
import tempfile
import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports (for Docker deployment)
# In Docker, all files are in /app, so we don't need parent directories
BASE_PATH = Path(__file__).parent
sys.path.insert(0, str(BASE_PATH))

# Import OCR modules
try:
    from paragraph_ocr import ocr_paragraph, correct_medical_terms
    from inference import load_model as load_trocr_model, ocr_image
except ImportError as e:
    logger.warning(f"Could not import English OCR modules: {e}")
    ocr_paragraph = None
    load_trocr_model = None

try:
    # Fix for PyTorch 2.6+ weights_only issue
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    
    from ultralytics import YOLO
    from model import Model
    from dataset import NormalizePAD
    from utils import CTCLabelConverter, AttnLabelConverter
except ImportError as e:
    logger.warning(f"Could not import Urdu OCR modules: {e}")
    YOLO = None
    Model = None

# FastAPI app
app = FastAPI(
    title="OCR Backend API",
    description="FastAPI backend for TrOCR-LoRA (English) and UTRNet + YOLO (Urdu) OCR",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
english_model = None
english_processor = None
english_device = None
urdu_detection_model = None
urdu_recognition_model = None
urdu_converter = None
urdu_opt = None
urdu_device = None

# Model paths (relative to ocr-backend directory)
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent


TROCR_CHECKPOINT = BASE_DIR / "models" / "trocr" / "trocr-lora-unified"
URDU_DETECTION_MODEL = BASE_DIR / "models" / "yolo" / "yolov8m_UrduDoc.pt"
URDU_RECOGNITION_MODEL = BASE_DIR / "models" / "utrnet" / "best_accuracy.pth"
URDU_GLYPHS_FILE = BASE_DIR / "UrduGlyphs.txt"

# Detection model path for English OCR (YOLO)
ENGLISH_DETECTION_MODEL = URDU_DETECTION_MODEL  # Can use same YOLO model for English detection


class OCRRequest(BaseModel):
    image: str  # base64 encoded image
    language: str = "english"  # "english" or "urdu"
    medical_corrections: bool = False  # Only for English
    visualize: bool = False  # Return visualization for English OCR


class OCRResponse(BaseModel):
    text: str
    visualization: Optional[str] = None  # base64 encoded visualization image


def validate_base64_image(base64_string: str) -> bool:
    """Validate if string is a valid base64 encoded image"""
    try:
        # Try to decode
        image_data = base64.b64decode(base64_string)
        # Try to open as image
        Image.open(BytesIO(image_data))
        return True
    except Exception:
        return False


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def process_english_ocr(image_path: str, medical_corrections: bool = False, visualize: bool = False) -> tuple[str, Optional[str]]:
    """
    Process English OCR using TrOCR-LoRA with paragraph processing
    
    Args:
        image_path: Path to input image
        medical_corrections: Apply medical term corrections
        visualize: Generate and return visualization
    
    Returns:
        Tuple of (extracted_text, visualization_base64 or None)
    """
    global english_model, english_processor, english_device
    
    try:
        # Check if models are loaded
        if english_model is None or english_processor is None:
            logger.error("English OCR models not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="English OCR models not loaded"
            )
        
        # Use paragraph_ocr function if available
        if ocr_paragraph is not None:
            # Create temporary visualization path if needed
            vis_path = None
            if visualize:
                vis_path = tempfile.mktemp(suffix='.png')
            
            # Run paragraph OCR
            result_text = ocr_paragraph(
                image_path=image_path,
                detection_model_path=str(ENGLISH_DETECTION_MODEL),
                trocr_checkpoint=str(TROCR_CHECKPOINT),
                conf_threshold=0.2,
                line_threshold=20,
                max_tokens=512,
                num_beams=10,
                save_visualization=vis_path,
                use_medical_corrections=medical_corrections
            )
            
            # Load visualization if generated
            vis_base64 = None
            if visualize and vis_path and os.path.exists(vis_path):
                vis_image = Image.open(vis_path)
                vis_base64 = image_to_base64(vis_image)
                # Clean up
                try:
                    os.unlink(vis_path)
                except:
                    pass
            
            return result_text, vis_base64
        else:
            # Fallback to simple OCR
            image = Image.open(image_path).convert('RGB')
            pixel_values = english_processor(image, return_tensors="pt").pixel_values.to(english_device)
            
            with torch.no_grad():
                generated_ids = english_model.generate(
                    pixel_values,
                    max_new_tokens=512,
                    num_beams=10,
                    early_stopping=True,
                    pad_token_id=english_processor.tokenizer.pad_token_id,
                    eos_token_id=english_processor.tokenizer.eos_token_id,
                )
            
            text = english_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Apply medical corrections if requested
            if medical_corrections:
                text = correct_medical_terms(text)
            
            return text, None
            
    except Exception as e:
        logger.error(f"Error in English OCR: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"English OCR processing failed: {str(e)}"
        )


def process_urdu_ocr(image_path: str) -> str:
    """
    Process Urdu OCR using UTRNet + YOLO
    
    Args:
        image_path: Path to input image
    
    Returns:
        Extracted text
    """
    global urdu_detection_model, urdu_recognition_model, urdu_converter, urdu_opt, urdu_device
    
    try:
        # Check if models are loaded
        if urdu_detection_model is None or urdu_recognition_model is None:
            logger.error("Urdu OCR models not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Urdu OCR models not loaded"
            )
        
        # Step 1: Detect text regions
        logger.info("Detecting Urdu text regions...")
        input_image = Image.open(image_path).convert('RGB')
        
        detection_results = urdu_detection_model.predict(
            source=input_image,
            conf=0.2,
            imgsz=1280,
            save=False,
            nms=True,
            device=urdu_device
        )
        
        bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
        bounding_boxes.sort(key=lambda x: x[1])  # Sort by y-coordinate
        
        if len(bounding_boxes) == 0:
            logger.warning("No text regions detected")
            return ""
        
        logger.info(f"Detected {len(bounding_boxes)} text regions")
        
        # Step 2: Recognize text in each region
        all_text = []
        for idx, bbox in enumerate(bounding_boxes):
            try:
                # Crop text region
                x1, y1, x2, y2 = bbox
                width, height = input_image.size
                x1 = max(0, int(x1) - 5)
                y1 = max(0, int(y1) - 5)
                x2 = min(width, int(x2) + 5)
                y2 = min(height, int(y2) + 5)
                
                cropped_image = input_image.crop((x1, y1, x2, y2))
                
                # Preprocess for UTRNet
                if urdu_opt.rgb:
                    img1 = cropped_image.convert('RGB')
                    img = img1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                else:
                    img1 = cropped_image.convert('L')
                    img = img1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                
                # Resize while maintaining aspect ratio
                w, h = img.size
                ratio = w / float(h)
                if math.ceil(urdu_opt.imgH * ratio) > urdu_opt.imgW:
                    resized_w = urdu_opt.imgW
                else:
                    resized_w = math.ceil(urdu_opt.imgH * ratio)
                
                img = img.resize((resized_w, urdu_opt.imgH), Image.Resampling.BICUBIC)
                
                # Normalize and pad
                transform = NormalizePAD((1, urdu_opt.imgH, urdu_opt.imgW))
                img = transform(img)
                img = img.unsqueeze(0)
                img = img.to(urdu_device)
                
                # Run recognition
                urdu_recognition_model.eval()
                with torch.no_grad():
                    preds = urdu_recognition_model(img)
                    preds_size = torch.IntTensor([preds.size(1)] * 1)
                    _, preds_index = preds.max(2)
                    preds_str = urdu_converter.decode(preds_index.data.cpu(), preds_size.data.cpu())[0]
                
                if preds_str and preds_str.strip():
                    all_text.append(preds_str.strip())
                    
            except Exception as e:
                logger.warning(f"Error recognizing region {idx + 1}: {e}")
                continue
        
        # Combine all text
        combined_text = ' '.join(all_text)
        return combined_text
        
    except Exception as e:
        logger.error(f"Error in Urdu OCR: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Urdu OCR processing failed: {str(e)}"
        )


@app.on_event("startup")
async def load_models():
    """Load OCR models at startup"""
    global english_model, english_processor, english_device
    global urdu_detection_model, urdu_recognition_model, urdu_converter, urdu_opt, urdu_device
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting OCR Backend API on port {port}")
    
    # Load English TrOCR model
    logger.info("Loading English TrOCR-LoRA model...")
    try:
        if load_trocr_model is not None and TROCR_CHECKPOINT.exists():
            logger.info(f"Loading model from: {TROCR_CHECKPOINT}")
            english_model, english_processor, english_device = load_trocr_model(
                str(TROCR_CHECKPOINT),
                device=torch.device("cpu")
            )
            logger.info("✅ English TrOCR-LoRA model loaded successfully")
        else:
            logger.warning(f"TrOCR checkpoint not found at {TROCR_CHECKPOINT}")
            logger.warning("English OCR will use fallback method")
    except Exception as e:
        logger.error(f"Failed to load English TrOCR model: {e}", exc_info=True)
        logger.warning("Continuing without English model - API will return errors for English OCR")
        # Don't let model loading failure crash the server
        english_model = None
        english_processor = None
    
    # Load Urdu OCR models
    logger.info("Loading Urdu OCR models...")
    try:
        if YOLO is not None and URDU_DETECTION_MODEL.exists():
            # Load detection model (CPU only)
            urdu_device = torch.device("cpu")
            logger.info(f"Using device: {urdu_device}")
            
            urdu_detection_model = YOLO(str(URDU_DETECTION_MODEL))
            logger.info("✅ Urdu detection model (YOLO) loaded")
            
            # Load recognition model
            if URDU_RECOGNITION_MODEL.exists() and URDU_GLYPHS_FILE.exists():
                # Load character set
                with open(URDU_GLYPHS_FILE, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                    character_set = ''.join([str(elem).strip('\n') for elem in content]) + " "
                
                # Create model options
                class Opt:
                    def __init__(self, device, character_set):
                        self.Prediction = "CTC"
                        self.character = character_set
                        self.imgH = 32
                        self.imgW = 400
                        self.rgb = False
                        self.input_channel = 1
                        self.output_channel = 32  # For HRNet
                        self.hidden_size = 256
                        self.FeatureExtraction = "HRNet"
                        self.SequenceModeling = "DBiLSTM"
                        self.num_fiducial = 20
                        self.device = device
                
                urdu_opt = Opt(urdu_device, character_set)
                
                # Create converter
                urdu_converter = CTCLabelConverter(urdu_opt.character)
                urdu_opt.num_class = len(urdu_converter.character)
                
                # Load model
                urdu_recognition_model = Model(urdu_opt)
                urdu_recognition_model = urdu_recognition_model.to(urdu_device)
                urdu_recognition_model.load_state_dict(
                    torch.load(str(URDU_RECOGNITION_MODEL), map_location=urdu_device, weights_only=False)
                )
                urdu_recognition_model.eval()
                
                logger.info("✅ Urdu recognition model (UTRNet) loaded")
            else:
                logger.warning(f"UTRNet model not found at {URDU_RECOGNITION_MODEL}")
        else:
            logger.warning(f"YOLO detection model not found at {URDU_DETECTION_MODEL}")
    except Exception as e:
        logger.error(f"Failed to load Urdu OCR models: {e}", exc_info=True)
        logger.warning("Continuing without Urdu model - API will return errors for Urdu OCR")
        # Don't let model loading failure crash the server
        urdu_detection_model = None
        urdu_recognition_model = None
    
    logger.info("✅ Model loading complete. API is ready to accept requests.")
    logger.info(f"English OCR available: {english_model is not None}")
    logger.info(f"Urdu OCR available: {urdu_detection_model is not None and urdu_recognition_model is not None}")


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "OCR Backend API",
        "endpoints": {
            "health": "/health",
            "ocr": "/ocr",
            "docs": "/docs"
        }
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Health check endpoint for platform monitoring"""
    models_status = {
        "english": english_model is not None and english_processor is not None,
        "urdu": urdu_detection_model is not None and urdu_recognition_model is not None
    }
    
    return {
        "status": "healthy",
        "models": models_status,
        "ready": models_status["english"] or models_status["urdu"]
    }


@app.post("/ocr", response_model=OCRResponse, status_code=status.HTTP_200_OK)
async def ocr(request: OCRRequest):
    """
    Perform OCR on a base64-encoded image.
    
    - **image**: Base64-encoded image string
    - **language**: Language to use for OCR ("english" or "urdu"), defaults to "english"
    - **medical_corrections**: Apply medical term corrections (English only, default: false)
    - **visualize**: Return visualization image (English only, default: false)
    
    Returns the extracted text from the image.
    """
    # Validate image
    if not request.image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image field is required"
        )
    
    if not validate_base64_image(request.image):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data"
        )
    
    # Normalize language parameter
    language = request.language.lower() if request.language else "english"
    
    if language not in ["english", "urdu"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {language}. Use 'english' or 'urdu'"
        )
    
    # Validate medical_corrections and visualize flags
    if language != "english":
        if request.medical_corrections:
            logger.warning("medical_corrections is only available for English OCR")
        if request.visualize:
            logger.warning("visualize is only available for English OCR")
    
    # Save image to temporary file
    temp_image_path = None
    try:
        image = base64_to_image(request.image)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name, format='PNG')
            temp_image_path = tmp_file.name
        
        # Process OCR based on language
        if language == "english":
            text, visualization = process_english_ocr(
                temp_image_path,
                medical_corrections=request.medical_corrections,
                visualize=request.visualize
            )
        else:  # urdu
            text = process_urdu_ocr(temp_image_path)
            visualization = None
        
        return OCRResponse(text=text, visualization=visualization)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing OCR request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
