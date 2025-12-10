"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from functools import partial
import random as rnd
import re
import imgaug.augmenters as iaa
import numpy as np
from PIL import ImageFilter, Image
from timm.data import auto_augment

_OP_CACHE = {}

def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))

def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)

def motion_blur(img, k, **__):
    k = _get_param(k, img, 0.08, 3) | 1  # bin to odd values
    key = 'motion_blur_' + str(k)
    op = _get_op(key, lambda: iaa.MotionBlur(k))
    return Image.fromarray(op(image=np.asarray(img)))

def gaussian_noise(img, scale, **_):
    scale = _get_param(scale, img, 0.25) | 1  # bin to odd values
    key = 'gaussian_noise_' + str(scale)
    op = _get_op(key, lambda: iaa.AdditiveGaussianNoise(scale=scale))
    return Image.fromarray(op(image=np.asarray(img)))

def poisson_noise(img, lam, **_):
    lam = _get_param(lam, img, 0.2) | 1  # bin to odd values
    key = 'poisson_noise_' + str(lam)
    op = _get_op(key, lambda: iaa.AdditivePoissonNoise(lam))
    return Image.fromarray(op(image=np.asarray(img)))

def salt_and_pepper_noise(image, prob=0.05):
    if prob <= 0:
        return image
    arr = np.asarray(image)
    original_dtype = arr.dtype
    intensity_levels = 2 ** (arr[0, 0].nbytes * 8)
    min_intensity = 0
    max_intensity = intensity_levels - 1
    random_image_arr = np.random.choice([min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape)
    salt_and_peppered_arr = arr.astype(np.float) * random_image_arr
    salt_and_peppered_arr = np.nan_to_num(salt_and_peppered_arr, nan=max_intensity).astype(original_dtype)
    return Image.fromarray(salt_and_peppered_arr)

def random_border_crop(image):
    img_width,img_height = image.size
    crop_left = int(img_width * rnd.uniform(0.0, 0.025))
    crop_top = int(img_height * rnd.uniform(0.0, 0.075))            
    crop_right = int(img_width * rnd.uniform(0.975, 1.0))
    crop_bottom = int(img_height * rnd.uniform(0.925, 1.0))
    final_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return final_image

def random_resize(image):
    size = image.size
    new_size = [rnd.randint(int(0.5*size[0]), int(1.5*size[0])), rnd.randint(int(0.5*size[1]), int(1.5*size[1]))]
    reduce_factor = rnd.randint(1,4)
    new_size = tuple([int(x/reduce_factor) for x in new_size])
    final_image = image.resize(new_size)
    return final_image

# ============================================================================
# PHASE 2: Targeted Augmentations for Urdu OCR Error Patterns
# ============================================================================

def dot_jitter(image, prob=0.15):
    """
    Dot Jitter Augmentation: Add/subtract/move dots on Urdu characters.
    Applies slight blur to dots to simulate dot detection difficulty.
    This helps improve robustness to dot detection failures.
    
    Target errors: "سٹاف"→"شاف", "بحال"→"بجال", "ایچ"→"انچ"
    """
    if prob <= 0 or rnd.random() >= prob:
        return image
    # Apply slight blur to simulate dot detection difficulty
    # This makes dots slightly less distinct, training the model to be more robust
    radius = rnd.uniform(0.2, 0.4)  # Very slight blur
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def synthetic_dot_noise(image, prob=0.1):
    """
    Synthetic Dot Noise: Add/remove small dots randomly.
    This directly attacks dot confusion errors by training on dot variations.
    
    Target errors: Dot position confusion (ب vs پ), missing dots (ٹ→س)
    """
    if prob <= 0 or rnd.random() >= prob:
        return image
    try:
        import cv2
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        
        # Add or remove small dots (1-2 pixels)
        num_dots = rnd.randint(1, 3)
        for _ in range(num_dots):
            x = rnd.randint(0, w-1)
            y = rnd.randint(0, h-1)
            # Small region for dot manipulation
            y1, y2 = max(0, y-1), min(h, y+2)
            x1, x2 = max(0, x-1), min(w, x+2)
            
            if rnd.random() < 0.5:
                # Remove dot: set small region to background (255 for white background)
                arr[y1:y2, x1:x2] = 255
            else:
                # Add dot: set small region to foreground (0 for black text)
                arr[y1:y2, x1:x2] = 0
        
        return Image.fromarray(arr)
    except ImportError:
        # Fallback: just return original if cv2 not available
        return image

def horizontal_elastic_distortion(image, max_shift=2, prob=0.2):
    """
    Horizontal Elastic Distortion: Slight horizontal stretching/compression.
    Handles ligature breakage in compound words.
    
    Target errors: "پیرامیڈکس"→"پیر امیڈ کس", "رپورٹنگ" splitting
    """
    if prob <= 0 or rnd.random() >= prob:
        return image
    
    # Calculate shift amount before try/except
    shift_amount = rnd.uniform(-max_shift, max_shift)
    
    try:
        import cv2
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        
        # Apply slight horizontal elastic transform
        # Create a simple horizontal shear/warp
        # Create transformation matrix for horizontal shear
        M = np.float32([[1, shift_amount/w, 0], [0, 1, 0]])
        arr = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(arr)
    except ImportError:
        # Fallback: simple horizontal shift if cv2 not available
        if abs(shift_amount) > 0.5:
            arr = np.asarray(image)
            h, w = arr.shape[:2]
            shift_pixels = int(shift_amount)
            if shift_pixels > 0:
                arr = np.pad(arr, ((0, 0), (0, abs(shift_pixels))), mode='edge')[:, abs(shift_pixels):]
            elif shift_pixels < 0:
                arr = np.pad(arr, ((0, 0), (abs(shift_pixels), 0)), mode='edge')[:, :w]
            return Image.fromarray(arr)
        return image

def stroke_thickness_variation(image, prob=0.15):
    """
    Stroke Thickness Variation: Vary stroke thickness (thin/thicken).
    Handles font style variations and helps with shape-similar character confusion.
    
    Target errors: "ی" vs "پ", thin stroke detection
    """
    if prob <= 0 or rnd.random() >= prob:
        return image
    try:
        import cv2
        arr = np.asarray(image)
        
        # Erode (thin) or dilate (thicken) strokes
        kernel = np.ones((2, 2), np.uint8)
        if rnd.random() < 0.5:
            # Thin strokes (erode)
            arr = cv2.erode(arr, kernel, iterations=1)
        else:
            # Thicken strokes (dilate)
            arr = cv2.dilate(arr, kernel, iterations=1)
        
        return Image.fromarray(arr)
    except ImportError:
        # Fallback: return original if cv2 not available
        return image

def number_augmentation(image, prob=0.3):
    """
    Number-Specific Augmentation: Special augmentation for numeric strings.
    Applies more aggressive augmentation for numeric regions.
    
    Target errors: "1000"→"13202", severe number recognition failure
    """
    if prob <= 0 or rnd.random() >= prob:
        return image
    
    # Apply multiple augmentations specifically for numbers
    # Higher contrast variation
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(rnd.uniform(1.1, 1.3))  # Increase contrast
    
    # Slight rotation (numbers are more rotation-sensitive)
    if rnd.random() < 0.5:
        angle = rnd.uniform(-1, 1)  # Very slight rotation
        image = image.rotate(angle, fillcolor=255, resample=Image.BICUBIC)
    
    return image

def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return level,

_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    # 'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
]
#_RAND_TRANSFORMS.remove('SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    'GaussianNoise',
    'PoissonNoise'
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=4),
    'MotionBlur': partial(_level_to_arg, max=20),
    'GaussianNoise': partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise': partial(_level_to_arg, max=40)
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise
})

def rand_augment_transform(magnitude=5, num_layers=3):
    # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
    hparams = {
        'img_mean':128,
        # 'rotate_deg': 5,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.0,
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams, transforms=_RAND_TRANSFORMS)
    # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
    choice_weights = [1. / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)

# ============================================================================
# TEXT-LEVEL AUGMENTATION: Spacing and Number Variations
# ============================================================================

def augment_spacing(text, prob=0.4):
    """
    Augment spacing in text to handle spacing errors.
    
    Target errors:
    - "کارروایی" → "کار روایی" (missing space)
    - "کویقینی" → "کو یقینی" (missing space)
    - "اوراس" → "اور اس" (missing space)
    - "تھاکہ" → "تھا کہ" (missing space)
    
    Args:
        text: Input text string
        prob: Probability of applying augmentation
    
    Returns:
        Augmented text with spacing variations
    """
    if prob <= 0 or rnd.random() >= prob:
        return text
    
    # Common Urdu compound words that might have spacing issues
    # These patterns are based on observed errors in the training log
    spacing_patterns = [
        # Compound words that might be split - be conservative, only apply to known patterns
        (r'(\w{2,})(روایی)', r'\1 \2'),  # کارروایی → کار روایی (min 2 chars before)
        (r'(\w{2,})(یقینی)', r'\1 \2'),  # کویقینی → کو یقینی
        (r'(اور)(\w{2,})', r'\1 \2'),    # اوراس → اور اس (min 2 chars after)
        (r'(\w{2,})(کہ)', r'\1 \2'),     # تھاکہ → تھا کہ
        (r'(\w{1,})(پر)(\s|$)', r'\1 \2\3'),  # زپر → ز پر (with word boundary)
        (r'(\w{2,})(کی)(\s|$)', r'\1 \2\3'),  # کی variations
        (r'(\w{2,})(کے)(\s|$)', r'\1 \2\3'),  # کے variations
        (r'(\w{2,})(نے)(\s|$)', r'\1 \2\3'),  # نے variations
        # Additional Urdu-specific patterns based on observed errors
        (r'(جب)(کہ)', r'\1 \2'),         # جبکہ → جب کہ
        (r'(\w{2,})(گیے)', r'\1 \2'),    # ہوگیے → ہو گیے
        (r'(\w{2,})(جاییں)', r'\1 \2'),  # کیےجاییں → کیے جاییں
        (r'(\w{2,})(شہید)', r'\1 \2'),   # شہیدکیا → شہید کیا
        (r'(\w{2,})(احکامات)', r'\1 \2'), # کےاحکامات → کے احکامات
        (r'(\w{2,})(انٹریاں)', r'\1 \2'), # نےانٹریاں → نے انٹریاں
        (r'(\w{2,})(عکاسی)', r'\1 \2'),  # کیعکاسی → کی عکاسی
        (r'(\w{2,})(وجہ)', r'\1 \2'),    # کیوجہ → کی وجہ
        (r'(\w{2,})(بہتری)', r'\1 \2'),  # میںبہتری → میں بہتری
        (r'(\w{2,})(اضافہ)', r'\1 \2'),  # میںاضافہ → میں اضافہ
        (r'(\w{2,})(رپورٹ)', r'\1 \2'), # کیرپورٹ → کی رپورٹ
        (r'(\w{2,})(تعداد)', r'\1 \2'), # کیتعداد → کی تعداد
        (r'(\w{2,})(زیادہ)', r'\1 \2'), # سےزیادہ → سے زیادہ
        (r'(\w{2,})(دوران)', r'\1 \2'), # کیدوران → کے دوران
        (r'(\w{2,})(ہمراہ)', r'\1 \2'), # کےہمراہ → کے ہمراہ
        (r'(\w{2,})(ملکر)', r'\1 \2'),  # ملکر → مل کر
        (r'(\w{2,})(کرسکیں)', r'\1 \2'), # کرسکیں → کر سکیں
        # Handle "روای" vs "روایی" confusion - add missing ی
        (r'(\w{2,})(روای)(\s|$)', r'\1\2ی\3'),  # کارروای → کارروایی (add missing ی)
    ]
    
    # Sometimes remove spaces (opposite direction) - less aggressive
    if rnd.random() < 0.25:  # 25% chance to remove spaces instead
        # Remove spaces between certain patterns, but be conservative
        # Only remove if both parts are at least 2 characters
        text = re.sub(r'(\w{2,}) (\w{2,})', 
                     lambda m: m.group(1) + m.group(2) if rnd.random() < 0.15 else m.group(0), 
                     text)
    else:
        # Add spaces based on patterns - apply only one pattern per augmentation
        patterns_to_apply = [p for p in spacing_patterns if rnd.random() < 0.25]
        if patterns_to_apply:
            # Apply only one random pattern to avoid over-augmentation
            pattern, replacement = rnd.choice(patterns_to_apply)
            text = re.sub(pattern, replacement, text, count=1)  # Only first match
    
    return text

def augment_number_spacing(text, prob=0.5):
    """
    Augment spacing around numbers to handle number recognition errors.
    
    Target errors:
    - "1000" → "113" (number recognition failure)
    - ")اے ایف پی(" → ") اے ایف پی (" (punctuation spacing)
    - Numbers without proper spacing
    
    Args:
        text: Input text string
        prob: Probability of applying augmentation
    
    Returns:
        Augmented text with number spacing variations
    """
    if prob <= 0 or rnd.random() >= prob:
        return text
    
    # Add/remove spaces around numbers
    # Pattern: number followed by non-space character or vice versa
    if rnd.random() < 0.5:
        # Add space after numbers
        text = re.sub(r'(\d+)([^\d\s])', r'\1 \2', text)
    else:
        # Add space before numbers
        text = re.sub(r'([^\d\s])(\d+)', r'\1 \2', text)
    
    # Handle punctuation spacing around numbers
    # Add space around punctuation near numbers
    text = re.sub(r'([\)\]\}])(\d+)', r'\1 \2', text)  # )1000 → ) 1000
    text = re.sub(r'(\d+)([\(\[\{])', r'\1 \2', text)   # 1000( → 1000 (
    
    return text

def augment_punctuation_spacing(text, prob=0.4):
    """
    Augment spacing around punctuation marks.
    
    Target errors:
    - ")اے ایف پی(" → ") اے ایف پی (" (missing spaces around punctuation)
    - Punctuation attached to words
    
    Args:
        text: Input text string
        prob: Probability of applying augmentation
    
    Returns:
        Augmented text with punctuation spacing variations
    """
    if prob <= 0 or rnd.random() >= prob:
        return text
    
    # Punctuation marks that need spacing
    closing_punct = r'[\)\]\}]'  # Closing brackets
    opening_punct = r'[\(\[\{]'   # Opening brackets
    
    # Add space after closing punctuation if followed by letter/digit (not space or punctuation)
    # Pattern: )text → ) text
    # Use string concatenation to avoid f-string curly brace conflicts
    pattern1 = '(' + closing_punct + ')([^\\s\\)\\]\\}\\(\\[\\{\\.\\,\\:\\;])'
    text = re.sub(pattern1, r'\1 \2', text)
    
    # Add space before opening punctuation if preceded by letter/digit (not space or punctuation)
    # Pattern: text( → text (
    pattern2 = '([^\\s\\)\\]\\}\\(\\[\\{\\.\\,\\:\\;])(' + opening_punct + ')'
    text = re.sub(pattern2, r'\1 \2', text)
    
    # Sometimes remove spaces (to train on both variations) - less aggressive
    if rnd.random() < 0.15:  # 15% chance to remove spaces
        # Remove space after closing punctuation
        pattern3 = '(' + closing_punct + ') '
        text = re.sub(pattern3, r'\1', text)
        # Remove space before opening punctuation
        pattern4 = ' (' + opening_punct + ')'
        text = re.sub(pattern4, r'\1', text)
    
    return text

def augment_number_format(text, prob=0.3):
    """
    Augment number formats to improve number recognition.
    
    Target errors:
    - "1000" → "113" (misread digits)
    - Different number representations
    
    Note: This is a conservative augmentation - we don't want to change
    numbers too much as it might confuse the model. Instead, we focus on
    spacing around numbers which is more common.
    
    Args:
        text: Input text string
        prob: Probability of applying augmentation
    
    Returns:
        Augmented text with number format variations
    """
    if prob <= 0 or rnd.random() >= prob:
        return text
    
    # Very conservative: only add slight variations for multi-digit numbers
    # We don't want to change numbers too much as it might confuse the model
    def replace_number(match):
        num_str = match.group(0)
        # Only augment if it's a multi-digit number (3+ digits)
        # And only with very low probability to avoid breaking the data
        if len(num_str) >= 3 and rnd.random() < 0.08:  # Increased from 5% to 8% for better coverage
            # Very rarely, simulate a single digit misread (most common error)
            if rnd.random() < 0.4:  # Increased from 0.3 to 0.4
                # Replace one random digit with another (simulate misreading)
                # Common errors: 1000 → 113 (misread 0 as 1, or missing digits)
                pos = rnd.randint(0, len(num_str) - 1)
                # Simulate common OCR errors: 0→1, 1→3, 3→1, etc.
                digit_replacements = {
                    '0': ['1', '6', '8'],
                    '1': ['3', '7', '4'],
                    '3': ['1', '8'],
                    '6': ['0', '8'],
                    '8': ['0', '6', '3'],
                }
                current_digit = num_str[pos]
                if current_digit in digit_replacements:
                    new_digit = rnd.choice(digit_replacements[current_digit])
                else:
                    new_digit = rnd.choice('0123456789')
                return num_str[:pos] + new_digit + num_str[pos+1:]
            # Sometimes simulate missing digits (1000 → 100, 113, etc.)
            elif rnd.random() < 0.2 and len(num_str) >= 4:
                # Remove one digit (simulate missing digit error)
                pos = rnd.randint(0, len(num_str) - 1)
                return num_str[:pos] + num_str[pos+1:]
        return num_str
    
    # Replace numbers with variations (very conservatively)
    text = re.sub(r'\d{3,}', replace_number, text)  # Only numbers with 3+ digits
    
    return text

def apply_text_augmentation(text, spacing_prob=0.4, number_spacing_prob=0.5, 
                          punctuation_prob=0.4, number_format_prob=0.3):
    """
    Apply all text augmentations with specified probabilities.
    
    Args:
        text: Input text string
        spacing_prob: Probability of spacing augmentation
        number_spacing_prob: Probability of number spacing augmentation
        punctuation_prob: Probability of punctuation spacing augmentation
        number_format_prob: Probability of number format augmentation
    
    Returns:
        Augmented text
    """
    # Apply augmentations in sequence
    text = augment_spacing(text, prob=spacing_prob)
    text = augment_number_spacing(text, prob=number_spacing_prob)
    text = augment_punctuation_spacing(text, prob=punctuation_prob)
    text = augment_number_format(text, prob=number_format_prob)
    
    # Normalize multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text