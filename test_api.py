"""
Test script for OCR Backend API
Usage: python test_api.py <image_path> [language] [medical_corrections] [visualize]
"""

import sys
import base64
import requests
import json
import os
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_ocr_api(image_path: str, language: str = "english", 
                 medical_corrections: bool = False, 
                 visualize: bool = False,
                 api_url: str = "http://localhost:8000"):
    """Test the OCR API endpoint"""
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    print(f"📸 Loading image: {image_path}")
    
    # Encode image to base64
    try:
        image_base64 = encode_image_to_base64(image_path)
        print(f"✅ Image encoded to base64 ({len(image_base64)} characters)")
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return
    
    # Prepare request
    url = f"{api_url}/ocr"
    payload = {
        "image": image_base64,
        "language": language,
        "medical_corrections": medical_corrections,
        "visualize": visualize
    }
    
    print(f"\n🚀 Sending request to: {url}")
    print(f"   Language: {language}")
    print(f"   Medical corrections: {medical_corrections}")
    print(f"   Visualize: {visualize}")
    
    try:
        # Send request
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout for processing
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("✅ OCR RESULT:")
            print("="*60)
            print(result.get("text", ""))
            print("="*60)
            
            # Save visualization if provided
            if result.get("visualization"):
                vis_path = Path(image_path).stem + "_visualization.png"
                vis_data = base64.b64decode(result["visualization"])
                with open(vis_path, 'wb') as f:
                    f.write(vis_data)
                print(f"\n💾 Visualization saved to: {vis_path}")
            
            # Save text result
            output_path = Path(image_path).stem + "_ocr_result.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("text", ""))
            print(f"💾 Text result saved to: {output_path}")
            
        else:
            print(f"\n❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timed out (processing took too long)")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to {api_url}")
        print("   Make sure the server is running: uvicorn main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path> [language] [medical_corrections] [visualize]")
        print("\nExamples:")
        print("  python test_api.py ../61.jpg english")
        print("  python test_api.py ../61.jpg english true")
        print("  python test_api.py ../61.jpg english true true")
        print("  python test_api.py ../urdu-text-detection/test.jpg urdu")
        sys.exit(1)
    
    image_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "english"
    medical_corrections = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    visualize = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
    
    test_ocr_api(image_path, language, medical_corrections, visualize)

