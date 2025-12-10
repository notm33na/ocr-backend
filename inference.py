import os
import torch
import warnings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from PIL import Image
import argparse

# Suppress warnings about uninitialized weights (normal for LoRA models)
warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

def load_model(checkpoint_path="trocr-lora-base", device=None, base_model_name=None):
    """Load trained TrOCR model with LoRA adapter"""
    if device is None:
        # Use CPU for cloud deployment
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Loading model from: {checkpoint_path}")
    
    # Determine base model based on checkpoint path or explicit argument
    if base_model_name:
        model_name = base_model_name
    elif "unified" in checkpoint_path.lower() or "both" in checkpoint_path.lower():
        model_name = "microsoft/trocr-base-handwritten"
        print("📚 Detected unified model - using handwritten base")
    elif "handwritten" in checkpoint_path.lower():
        model_name = "microsoft/trocr-base-handwritten"
        print("📝 Detected handwritten model - using handwritten base")
    else:
        model_name = "microsoft/trocr-base-printed"
        print("📄 Using printed base model")
    
    base_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Load LoRA adapter
    try:
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print("✅ LoRA adapter loaded")
    except Exception as e:
        print(f"⚠️  Could not load as PEFT model: {e}")
        print("Trying to load as regular model...")
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
    
    model = model.to(device)
    model.eval()
    
    # Load processor
    if os.path.exists(os.path.join(checkpoint_path, "preprocessor_config.json")):
        processor = TrOCRProcessor.from_pretrained(checkpoint_path)
    else:
        processor = TrOCRProcessor.from_pretrained(model_name)
    
    # Set required token IDs
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    
    if hasattr(model.config, 'decoder'):
        if hasattr(model.config.decoder, 'decoder_start_token_id'):
            decoder_start_token_id = model.config.decoder.decoder_start_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = eos_token_id
                model.config.decoder.decoder_start_token_id = decoder_start_token_id
        else:
            decoder_start_token_id = eos_token_id
    else:
        decoder_start_token_id = eos_token_id
    
    model.config.pad_token_id = pad_token_id
    model.config.decoder_start_token_id = decoder_start_token_id
    
    return model, processor, device

def ocr_image(model, processor, device, image_path, target_height=256, max_new_tokens=512, num_beams=5):
    """Perform OCR on a single image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target height while keeping aspect ratio
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_width = int(target_height * aspect_ratio)
    image = image.resize((new_width, target_height), Image.Resampling.BILINEAR)
    
    # Process image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate text with parameters optimized for paragraphs
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    # Decode text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="OCR inference with trained TrOCR LoRA model")
    parser.add_argument("--checkpoint", type=str, default="trocr-lora-base",
                       help="Path to trained model checkpoint (default: trocr-lora-base, or use trocr-lora-unified for unified model)")
    parser.add_argument("--base-model", type=str, default=None, choices=["printed", "handwritten"],
                       help="Explicitly specify base model: 'printed' or 'handwritten' (auto-detected if not specified)")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to image file to OCR")
    parser.add_argument("--output", type=str, default=None,
                       help="Optional: Save OCR result to text file")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum number of tokens to generate (default: 512, use 1024+ for long paragraphs)")
    parser.add_argument("--num-beams", type=int, default=5,
                       help="Number of beams for beam search (default: 5, higher = better quality but slower)")
    parser.add_argument("--paragraph", action="store_true",
                       help="Optimize for paragraph text (sets max-tokens=1024)")
    
    args = parser.parse_args()
    
    # Auto-adjust for paragraphs
    max_tokens = 1024 if args.paragraph else args.max_tokens
    
    # Determine base model name if explicitly specified
    base_model_name = None
    if args.base_model:
        base_model_name = f"microsoft/trocr-base-{args.base_model}"
    
    # Load model
    model, processor, device = load_model(args.checkpoint, base_model_name=base_model_name)
    
    # Perform OCR
    print(f"\n📸 Processing image: {args.image}")
    if args.paragraph:
        print(f"📄 Paragraph mode: max_tokens={max_tokens}")
    try:
        text = ocr_image(model, processor, device, args.image, 
                        max_new_tokens=max_tokens, 
                        num_beams=args.num_beams)
        print(f"\n✅ OCR Result:")
        print(f"   {text}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\n💾 Saved to: {args.output}")
        
        return text
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()

