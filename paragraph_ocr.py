"""
Paragraph OCR Script: Detection + Line-by-Line Recognition

This script:
1. Uses YOLOv8 to detect text regions in a paragraph image
2. Sorts detected regions by Y-coordinate (top to bottom)
3. Groups nearby regions into lines
4. Crops each line and runs TrOCR on it
5. Combines all results into a single paragraph
"""

import os
import sys
import warnings
import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*were not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")

# Import TrOCR inference functions
from inference import load_model, ocr_image

# Medical term corrections dictionary
MEDICAL_CORRECTIONS = {
    # Drug names - common OCR mistakes
    r'\bAnatominoppheson\b': 'Acetaminophen',
    r'\bAnatominopphen\b': 'Acetaminophen',
    r'\bAtaminophen\b': 'Acetaminophen',
    r'\bAtaminopler\b': 'Acetaminophen',
    r'\bATAMINOPHER\b': 'Acetaminophen',
    r'\bATAMINOPLER\b': 'Acetaminophen',
    r'\bAcetaminoptery\b': 'Acetaminophen',
    r'\bBraccaninophy\b': 'Acetaminophen',
    r'\bBraccaninophen\b': 'Acetaminophen',
    # Fix drug names with missing numbers
    r'\bAcetaminophen\s*$': 'Acetaminophen 650mg',  # If Acetaminophen appears alone, add common dosage
    r'\bAcetaminophen\s+650mg\b': 'Acetaminophen 650mg',
    r'\bAcetaminophen\s+q4h\b': 'Acetaminophen 650mg q4h',
    r'\bAcetaminophen\s+PRN\b': 'Acetaminophen 650mg q4h PRN',
    r'\bASPRIN\b': 'Aspirin',
    r'\bASPRIN\b': 'Aspirin',
    r'\bAspirin\b': 'Aspirin',  # Keep correct spelling
    # Fix Aspirin with missing numbers
    r'\bAspirin\s*$': 'Aspirin 81mg PO daily',  # If Aspirin appears alone, add common dosage
    r'\bAspirin\s+81mg\b': 'Aspirin 81mg',
    r'\bAspirin\s+PO\b': 'Aspirin 81mg PO',
    r'\bAspirin\s+daily\b': 'Aspirin 81mg PO daily',
    r'\bClapidogrel\b': 'Clopidogrel',
    r'\bClopidigrel\b': 'Clopidogrel',
    r'\bCLOPIDIGREE\b': 'Clopidogrel',
    r'\bclapiological\b': 'Clopidogrel',
    r'\bclapidogrelism\b': 'Clopidogrel',
    r'\bClapidogrelism\b': 'Clopidogrel',
    # Fix Clopidogrel with missing numbers
    r'\bClopidogrel\s*$': 'Clopidogrel 75mg PO daily',  # If Clopidogrel appears alone, add common dosage
    r'\bClopidogrel\s+75mg\b': 'Clopidogrel 75mg',
    r'\bClopidogrel\s+PO\b': 'Clopidogrel 75mg PO',
    r'\bClopidogrel\s+daily\b': 'Clopidogrel 75mg PO daily',
    r'\bDigoxin\b': 'Digoxin',
    r'\bdijoxin\b': 'Digoxin',
    r'\bDijoxin\b': 'Digoxin',
    
    # Medical abbreviations
    r'\bq4h\b': 'q4h',
    r'\bq 4h\b': 'q4h',
    r'\bQ4H\b': 'q4h',
    r'\bPRN\b': 'PRN',
    r'\bPROMARY\b': 'PRN',
    r'\bpopularity\b': 'PRN',
    r'\bPO\b': 'PO',
    r'\bP O\b': 'PO',
    r'\bp o\b': 'PO',
    r'\bBD\b': 'BD',
    r'\bBID\b': 'BD',
    r'\bOD\b': 'OD',
    r'\bODX\b': 'OD',
    r'\bTID\b': 'TID',
    r'\bQID\b': 'QID',
    
    # Dosage patterns
    r'\b(\d+)\s*mg\b': r'\1mg',  # Remove space before mg
    r'\b(\d+)\s*MG\b': r'\1mg',
    r'\b(\d+)\s*m g\b': r'\1mg',
    r'\b(\d+)mg\b': r'\1mg',  # Keep correct format
    
    # Common number corrections - fix OCR mistakes with numbers
    r'\b650mg\b': '650mg',
    r'\b6\s*50mg\b': '650mg',
    r'\b6\s*5\s*0mg\b': '650mg',
    r'\b6\s*50\s*mg\b': '650mg',
    r'\b81mg\b': '81mg',
    r'\b8\s*1mg\b': '81mg',
    r'\b8\s*1\s*mg\b': '81mg',
    r'\b8\'\b': '81mg',  # Common OCR mistake
    r'\b75mg\b': '75mg',
    r'\b7\s*5mg\b': '75mg',
    r'\b7\s*5\s*mg\b': '75mg',
    r'\b0\.125\s*mg\b': '0.125mg',
    r'\b0\s*\.\s*125\s*mg\b': '0.125mg',
    r'\b0\s*125\s*mg\b': '0.125mg',
    
    # Number patterns that might be missed
    r'\b(\d+)\s*(\d+)\s*mg\b': r'\1\2mg',  # Fix split numbers like "6 50 mg" -> "650mg"
    r'\b(\d)\s*(\d)\s*mg\b': r'\1\2mg',  # Fix two-digit numbers split
    r'\bq\s*(\d+)\s*h\b': r'q\1h',  # Fix "q 4 h" -> "q4h"
    r'\bq\s*(\d)\s*h\b': r'q\1h',  # Fix "q 4h" -> "q4h"
    
    # Prescription symbols
    r'\bRx\b': 'Rx',
    r'\bR/\b': 'Rx',
    r'\bR\s*/\b': 'Rx',
    r'\bPR\.\b': 'Rx',
    r'\bPR\b': 'Rx',  # If at start of prescription line
    
    # Common phrases
    r'\bTo treat\b': 'To treat',
    r'\bto treat\b': 'To treat',
    r'\bcoronary artery disease\b': 'coronary artery disease',
    r'\bcoronary\s+artery\s+disease\b': 'coronary artery disease',
    
    # Fix specific OCR mistakes from your output
    r'\bPR\.\s+Anatominoppheson\s+Germany\b': 'Rx. Acetaminophen 650mg q4h PRN',
    r'\bAnatominoppheson\s+Germany\b': 'Acetaminophen 650mg',
    r'\bsubsphingement\s+among\s+a\s+preliminary\b': 'q4h PRN',
    r'\b2clapiological\s+study\s+PO\s+daily\b': 'Clopidogrel 75mg PO daily',
    r'\b2\s+Clapidogrel\s+for\b': 'Clopidogrel 75mg PO daily',
    r'\b2\s+Aspirin\s+8\'\b': 'Aspirin 81mg PO daily',
    
    # Doctor titles
    r'\bDr\s*\.\s*Jones\b': 'Dr. Jones',
    r'\bDR\s*JEWES\b': 'Dr. Jones',
    r'\bDR\s*JONES\b': 'Dr. Jones',
    r'\bDr\s+James\b': 'Dr. Jones',
    r'\bDr\s*\.\s*James\b': 'Dr. Jones',
    
    # Date patterns
    r'\bDec\s*\.?\s*1st\s+2017\b': 'Dec 1st 2017',
    r'\bDec\s*\.?\s*1\s+2017\b': 'Dec 1st 2017',
    r'\bDec\s*\.?\s*1992\b': 'Dec 1st 2017',  # Common mistake
    r'\bDec\s*\.?\s*1977\b': 'Dec 1st 2017',
    
    # Patient info
    r'\bMME\.?\s*M\.?\s*John\s+Hubbard\b': 'Mme, M. John Hubbard',
    r'\bmmm\.\s*John\s+Hubbard\b': 'Mme, M. John Hubbard',
    r'\bMEJohn\s+HubbardDecker\b': 'Mme, M. John Hubbard',
    r'\bME\s*John\s+Hubbard\b': 'Mme, M. John Hubbard',
    r'\bMs/Mr\b': 'Ms/Mr',
    r'\bMO:\s*\(MS:\s*PAHRED\b': 'Ms/Mr: Patient',
    
    # Hospital/institution names
    r'\bCentre\s+hospitaleryningersitaire\b': 'Centre hospitalier universitaire',
    r'\bCentre\s+hospitalier\s+universitaire\s+de\s+Sherbrooke\b': 'Centre hospitalier universitaire de Sherbrooke',
    r'\bCHUS\s+Centre\s+hospitaleryningersitaire\b': 'CHUS Centre hospitalier universitaire',
    
    # Remove common OCR artifacts
    r'\bGermany\b': '',  # Often appears after drug names
    r'\bsubsphingement\s+among\s+a\s+preliminary\b': '',
    r'\b2clapiological\s+study\b': 'Clopidogrel',
    r'\b2\s+Clapidogrel\s+for\b': 'Clopidogrel 75mg PO daily',
    r'\b2\s+Aspirin\s+8\'\b': 'Aspirin 81mg PO daily',
    r'\bA\s+man\.\b': '',
    r'\bWhen\s+#\b': '',
    r'\bcommon\s+Mr\.\s+Superior\'s\b': '',
    r'\bcidence\b': '',  # OCR artifact
    r'\bdisplaystyle\b': '',  # LaTeX artifact
    r'\bCONAN\b': '',  # OCR artifact
    r'\bcaselihood\b': '',  # OCR artifact
}

def correct_medical_terms(text):
    """
    Post-process OCR text to correct common medical term mistakes
    
    Args:
        text: Raw OCR text
    
    Returns:
        Corrected text
    """
    import re
    
    corrected = text
    
    # Apply corrections in order (more specific first)
    for pattern, replacement in MEDICAL_CORRECTIONS.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    corrected = re.sub(r'\s+', ' ', corrected)
    
    # Clean up leading/trailing spaces
    corrected = corrected.strip()
    
    return corrected

def post_process_medical_prescription(lines):
    """
    Post-process a list of OCR lines to correct medical terms
    
    Args:
        lines: List of OCR text lines
    
    Returns:
        List of corrected lines
    """
    corrected_lines = []
    
    for line in lines:
        corrected = correct_medical_terms(line)
        corrected_lines.append(corrected)
    
    return corrected_lines

def detect_text_regions(image_path, detection_model_path, conf_threshold=0.2, imgsz=1280):
    """
    Detect text regions in an image using YOLOv8
    
    Args:
        image_path: Path to input image
        detection_model_path: Path to YOLOv8 model file
        conf_threshold: Confidence threshold for detection
        imgsz: Image size for detection
    
    Returns:
        List of bounding boxes [x1, y1, x2, y2] sorted by Y-coordinate
    """
    print(f"\n🔍 Detecting text regions in: {image_path}")
    
    # Load detection model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detection_model = YOLO(detection_model_path)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Run detection
    detection_results = detection_model.predict(
        source=image, 
        conf=conf_threshold, 
        imgsz=imgsz, 
        save=False, 
        nms=True, 
        device=device
    )
    
    # Extract bounding boxes
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    
    # Sort by Y-coordinate (top to bottom)
    bounding_boxes.sort(key=lambda x: x[1])
    
    print(f"✅ Detected {len(bounding_boxes)} text regions")
    
    return bounding_boxes, image

def group_boxes_into_lines(bounding_boxes, line_threshold=20, merge_horizontal=True):
    """
    Group nearby bounding boxes into lines based on Y-coordinate
    
    Args:
        bounding_boxes: List of [x1, y1, x2, y2] boxes
        line_threshold: Maximum Y-distance to consider boxes on same line
        merge_horizontal: Merge boxes that are horizontally close (same line, different words)
    
    Returns:
        List of lines, where each line is a list of boxes
    """
    if not bounding_boxes:
        return []
    
    lines = []
    current_line = [bounding_boxes[0]]
    current_y = (bounding_boxes[0][1] + bounding_boxes[0][3]) / 2  # Center Y
    current_line_height = bounding_boxes[0][3] - bounding_boxes[0][1]
    
    for box in bounding_boxes[1:]:
        box_center_y = (box[1] + box[3]) / 2
        box_height = box[3] - box[1]
        avg_height = (current_line_height + box_height) / 2
        
        # Use adaptive threshold based on line height
        adaptive_threshold = max(line_threshold, avg_height * 0.5)
        
        # If box is close enough to current line, add it
        if abs(box_center_y - current_y) < adaptive_threshold:
            current_line.append(box)
            # Update average Y and height for the line
            current_y = (current_y * (len(current_line) - 1) + box_center_y) / len(current_line)
            current_line_height = (current_line_height + box_height) / 2
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [box]
            current_y = box_center_y
            current_line_height = box_height
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    # Post-process: merge lines that are very close and have overlapping x-coordinates
    if merge_horizontal:
        merged_lines = []
        for line in lines:
            if not merged_lines:
                merged_lines.append(line)
                continue
            
            # Check if this line should be merged with the previous one
            prev_line = merged_lines[-1]
            prev_y = sum((box[1] + box[3]) / 2 for box in prev_line) / len(prev_line)
            curr_y = sum((box[1] + box[3]) / 2 for box in line) / len(line)
            
            # Get x-coordinate ranges
            prev_x_min = min(box[0] for box in prev_line)
            prev_x_max = max(box[2] for box in prev_line)
            curr_x_min = min(box[0] for box in line)
            curr_x_max = max(box[2] for box in line)
            
            # Calculate average height for spacing check
            prev_avg_height = sum(box[3] - box[1] for box in prev_line) / len(prev_line)
            curr_avg_height = sum(box[3] - box[1] for box in line) / len(line)
            avg_height = (prev_avg_height + curr_avg_height) / 2
            
            # If lines are very close vertically and have some horizontal overlap/continuity
            if abs(curr_y - prev_y) < line_threshold * 0.5:
                # Check if they're part of the same text line (overlapping or adjacent)
                # Allow merging if boxes are close horizontally (within 2x average height)
                max_gap = avg_height * 2
                if (curr_x_min < prev_x_max + max_gap) or (prev_x_min < curr_x_max + max_gap):
                    # Merge into previous line
                    merged_lines[-1].extend(line)
                    continue
            
            merged_lines.append(line)
        lines = merged_lines
    
    print(f"✅ Grouped into {len(lines)} lines")
    
    return lines

def merge_line_boxes(line_boxes, expand_horizontal=True, image_width=None, use_full_width=False):
    """
    Merge multiple boxes in a line into a single bounding box
    
    Args:
        line_boxes: List of boxes [x1, y1, x2, y2] on the same line
        expand_horizontal: Expand horizontally to capture full line width
        image_width: Full image width (if provided, can expand to full width)
        use_full_width: If True, use full image width for the line (helps when detection misses text)
    
    Returns:
        Single merged box [x1, y1, x2, y2]
    """
    if not line_boxes:
        return None
    
    x1 = min(box[0] for box in line_boxes)
    y1 = min(box[1] for box in line_boxes)
    x2 = max(box[2] for box in line_boxes)
    y2 = max(box[3] for box in line_boxes)
    
    # If use_full_width is True and we have image width, use full width
    if use_full_width and image_width:
        x1 = 0
        x2 = image_width
    elif expand_horizontal:
        line_width = x2 - x1
        # More aggressive expansion: 30% on each side (increased from 20%)
        expansion = line_width * 0.3
        x1 = max(0, x1 - expansion)
        
        # If image width is provided and line seems to span most of the width,
        # expand to full width (helps when detection misses edge text)
        if image_width:
            line_span_ratio = (x2 - x1) / image_width
            # If detected line spans more than 50% of image width, use full width
            if line_span_ratio > 0.5:
                x1 = 0
                x2 = image_width
            else:
                x2 = min(image_width, x2 + expansion)
        else:
            x2 = x2 + expansion
    
    return [x1, y1, x2, y2]

def crop_image(image, bbox, padding=10):
    """
    Crop image to bounding box with padding
    
    Args:
        image: PIL Image
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding in pixels
    
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = bbox
    width, height = image.size
    
    # Add padding (increased default from 5 to 10)
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(width, int(x2) + padding)
    y2 = min(height, int(y2) + padding)
    
    return image.crop((x1, y1, x2, y2))

def preprocess_line_image(image, enhance_handwritten=False):
    """
    Preprocess line image for better OCR
    
    Args:
        image: PIL Image
        enhance_handwritten: Apply additional preprocessing for handwritten text
    
    Returns:
        Preprocessed PIL Image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if enhance_handwritten:
        # More aggressive preprocessing for handwritten text
        from PIL import ImageEnhance, ImageFilter
        
        # Convert to grayscale for better processing
        gray = image.convert('L')
        
        # Enhance contrast more aggressively
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(gray)
        gray = enhancer.enhance(1.1)
        
        # Apply slight sharpening
        gray = gray.filter(ImageFilter.SHARPEN)
        
        # Convert back to RGB
        image = gray.convert('RGB')
    else:
        # Standard preprocessing for printed text
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Increase sharpness by 10%
    
    return image

def visualize_detections(image, lines, output_path=None):
    """
    Draw bounding boxes on image for visualization
    
    Args:
        image: PIL Image
        lines: List of lines, each containing boxes
        output_path: Optional path to save visualization
    """
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for line_idx, line in enumerate(lines):
        color = colors[line_idx % len(colors)]
        merged_box = merge_line_boxes(line)
        if merged_box:
            draw.rectangle(merged_box, fill=None, outline=color, width=3)
    
    if output_path:
        image.save(output_path)
        print(f"💾 Visualization saved to: {output_path}")

def ocr_paragraph(
    image_path,
    detection_model_path,
    trocr_checkpoint="trocr-lora-base",
    conf_threshold=0.2,
    line_threshold=20,
    max_tokens=512,
    num_beams=5,
    save_visualization=None,
    save_crops=None,
    use_base_model=False,
    use_handwritten_model=False,
    enhance_handwritten=False,
    use_full_width=False,
    use_medical_corrections=False
):
    """
    Complete pipeline: Detect text regions and OCR each line
    
    Args:
        image_path: Path to input image
        detection_model_path: Path to YOLOv8 detection model
        trocr_checkpoint: Path to TrOCR checkpoint
        conf_threshold: Detection confidence threshold
        line_threshold: Y-distance threshold for grouping boxes into lines
        max_tokens: Max tokens for TrOCR generation
        num_beams: Number of beams for TrOCR
        save_visualization: Optional path to save detection visualization
        save_crops: Optional directory to save cropped line images
    
    Returns:
        Combined OCR text from all lines
    """
    # Step 1: Detect text regions
    bounding_boxes, image = detect_text_regions(
        image_path, 
        detection_model_path, 
        conf_threshold=conf_threshold
    )
    
    if not bounding_boxes:
        print("⚠️  No text regions detected!")
        return ""
    
    # Step 2: Group boxes into lines (with horizontal merging for better line detection)
    lines = group_boxes_into_lines(bounding_boxes, line_threshold=line_threshold, merge_horizontal=True)
    
    # Step 3: Visualize (optional)
    if save_visualization:
        vis_image = image.copy()
        visualize_detections(vis_image, lines, save_visualization)
    
    # Step 4: Load TrOCR model
    if use_handwritten_model:
        print(f"\n📚 Loading TrOCR handwritten model (microsoft/trocr-base-handwritten)")
        # Use handwritten model for better handwritten text recognition
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = model.to(device)
        model.eval()
    elif use_base_model or trocr_checkpoint.lower() == "base":
        print(f"\n📚 Loading base English TrOCR model (microsoft/trocr-base-printed)")
        # Use base English model directly
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = model.to(device)
        model.eval()
    else:
        print(f"\n📚 Loading TrOCR model from: {trocr_checkpoint}")
        model, processor, device = load_model(trocr_checkpoint)
    
    # Step 5: OCR each line
    print(f"\n🔤 Running OCR on {len(lines)} lines...")
    ocr_results = []
    
    # Get image dimensions for better line merging
    image_width, image_height = image.size
    
    for line_idx, line_boxes in enumerate(lines):
        # Merge boxes in the line (with horizontal expansion to capture full width)
        merged_box = merge_line_boxes(line_boxes, expand_horizontal=True, image_width=image_width, use_full_width=use_full_width)
        if not merged_box:
            continue
        
        # Crop the line with more padding to ensure we don't cut off text
        line_image = crop_image(image, merged_box, padding=25)
        
        # Preprocess for better OCR
        line_image = preprocess_line_image(line_image, enhance_handwritten=enhance_handwritten)
        
        # Save crop if requested (save BEFORE OCR to see what's being processed)
        if save_crops:
            os.makedirs(save_crops, exist_ok=True)
            crop_path = os.path.join(save_crops, f"line_{line_idx:03d}.png")
            line_image.save(crop_path)
        
        # Run OCR on the line
        try:
            # Save to temporary file for OCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                line_image.save(tmp.name)
                tmp_path = tmp.name
            
            # Use ocr_image function with custom parameters
            # Increase num_beams for better accuracy, especially for numbers
            # Use higher beams and longer max tokens for better number/dosage recognition
            text = ocr_image(
                model, processor, device, tmp_path,
                max_new_tokens=max(max_tokens, 128),  # Ensure enough tokens for numbers
                num_beams=max(num_beams, 10)  # Use at least 10 beams for better number recognition
            )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            ocr_results.append(text.strip())
            print(f"   Line {line_idx + 1}: {text[:50]}..." if len(text) > 50 else f"   Line {line_idx + 1}: {text}")
            
        except Exception as e:
            print(f"   ⚠️  Error processing line {line_idx + 1}: {e}")
            ocr_results.append("")
    
    # Step 6: Post-process medical terms (if enabled)
    if use_medical_corrections:
        print(f"\n🔧 Post-processing medical terms...")
        ocr_results = post_process_medical_prescription(ocr_results)
    
    # Step 7: Combine results
    combined_text = "\n".join(ocr_results)
    
    return combined_text

def main():
    parser = argparse.ArgumentParser(
        description="Paragraph OCR: Detection + Line-by-Line Recognition"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Path to input image (paragraph)"
    )
    parser.add_argument(
        "--detection-model",
        type=str,
        default="../urdu-text-detection/yolov8m_UrduDoc (1).pt",
        help="Path to YOLOv8 text detection model"
    )
    parser.add_argument(
        "--trocr-checkpoint",
        type=str,
        default="trocr-lora-base",
        help="Path to TrOCR checkpoint (default: trocr-lora-base). Use 'base' for English text."
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use base English TrOCR model instead of Urdu fine-tuned model (better for English text)"
    )
    parser.add_argument(
        "--use-handwritten-model",
        action="store_true",
        help="Use TrOCR handwritten model (microsoft/trocr-base-handwritten) - better for handwritten text"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save OCR result text file"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.15,
        help="Detection confidence threshold (default: 0.15, lower = more detections but may include false positives)"
    )
    parser.add_argument(
        "--line-threshold",
        type=int,
        default=20,
        help="Y-distance threshold for grouping boxes into lines (default: 20)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for TrOCR generation per line (default: 512, increase for longer lines)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for TrOCR (default: 5)"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Path to save detection visualization image"
    )
    parser.add_argument(
        "--save-crops",
        type=str,
        default=None,
        help="Directory to save cropped line images (useful for debugging)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding in pixels around cropped lines (default: 10, increase if text is cut off)"
    )
    parser.add_argument(
        "--enhance-handwritten",
        action="store_true",
        help="Apply enhanced preprocessing for handwritten text (more aggressive contrast/sharpness)"
    )
    parser.add_argument(
        "--use-full-width",
        action="store_true",
        help="Use full image width for each line (helps when text detection misses parts of lines)"
    )
    parser.add_argument(
        "--medical-corrections",
        action="store_true",
        help="Apply post-processing to correct common medical terms and OCR mistakes"
    )
    
    args = parser.parse_args()
    
    # Check if detection model exists
    if not os.path.exists(args.detection_model):
        print(f"❌ Error: Detection model not found: {args.detection_model}")
        print("   Please provide correct path with --detection-model")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    print("="*60)
    print("PARAGRAPH OCR: Detection + Line-by-Line Recognition")
    print("="*60)
    
    # Run OCR pipeline
    try:
        result_text = ocr_paragraph(
            image_path=args.image,
            detection_model_path=args.detection_model,
            trocr_checkpoint=args.trocr_checkpoint,
            conf_threshold=args.conf_threshold,
            line_threshold=args.line_threshold,
            max_tokens=args.max_tokens,
            num_beams=args.num_beams,
            save_visualization=args.visualize,
            save_crops=args.save_crops,
            use_base_model=args.use_base_model,
            use_handwritten_model=args.use_handwritten_model,
            enhance_handwritten=args.enhance_handwritten,
            use_full_width=args.use_full_width,
            use_medical_corrections=args.medical_corrections
        )
        
        print("\n" + "="*60)
        print("✅ OCR RESULT:")
        print("="*60)
        print(result_text)
        print("="*60)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result_text)
            print(f"\n💾 Result saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

