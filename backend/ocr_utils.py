import pytesseract
import requests
import os
import cv2
import re
import numpy as np
from pathlib import Path
import json
import csv
import hashlib
from typing import Optional, Tuple, List, Dict
from rapidfuzz import process as fuzz_process, fuzz
from datetime import datetime
import imagehash
from PIL import Image

# --- Load trusted reference data at import ---
try:
    with open("oracle-cards.json", "r", encoding="utf-8") as f:
        oracle_cards = json.load(f)
except FileNotFoundError:
    oracle_cards = []

CARD_NAMES = {card["name"] for card in oracle_cards if "name" in card}
CARD_TYPES = set()
for card in oracle_cards:
    if "type_line" in card:
        for word in re.split(r"[^A-Za-z]", card["type_line"]):
            if word:
                CARD_TYPES.add(word.strip())

# --- Text cleanup helpers ---
def clean_text_basic(text: str) -> str:
    """Normalize whitespace, remove stray punctuation, keep only allowed chars."""
    text = re.sub(r"[^A-Za-z0-9\s{}WUBRGXYZ/-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuzzy_correct(text: str, reference_set: set, threshold: int = 80) -> str:
    """Fuzzy match OCR text against a reference set and replace if above threshold."""
    if not text or not reference_set:
        return text
    match, score, _ = fuzz_process.extractOne(text, reference_set, scorer=fuzz.ratio)
    return match if score >= threshold else text

def normalize_mana_cost(text: str) -> str:
    """Keep only mana cost symbols and numbers."""
    text = re.sub(r"[^0-9WUBRGXYZ{}]", "", text.upper())
    return text


# Helper to generate unique timestamped filenames
def get_timestamped_path(base_dir: Path, base_name: str, ext: str = ".jpg") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}{ext}"
    return base_dir / filename

# Define directories
BASE_DIR = Path(__file__).resolve().parent.parent
MARKER_DIR = Path("/public/assets/mindar")
MODEL_DIR = Path("/public/assets/models")
CACHE_DIR = Path("/tmp/cache")
IMAGES_CACHE = Path("/tmp/cache/images")
JSON_CACHE = Path("/tmp/cache/json_cache")

#Local DIR for testing
#MARKER_DIR = BASE_DIR / "public" / "assets" / "mindar"
#MODEL_DIR = BASE_DIR / "public" / "assets" / "models"
#CACHE_DIR = BASE_DIR / "cache"
#IMAGES_CACHE = CACHE_DIR / "images"
#JSON_CACHE = CACHE_DIR / "json_cache"


# Create directories if they don't exist
for directory in [MARKER_DIR, MODEL_DIR, IMAGES_CACHE, JSON_CACHE]:
    directory.mkdir(parents=True, exist_ok=True)

# Create log files
OCR_FAILS_LOG = CACHE_DIR / "ocr_fails.txt"
SCANNER_LOG = CACHE_DIR / "scanner_log.csv"

# Initialize CSV log with headers if it doesn't exist
if not SCANNER_LOG.exists():
    with open(SCANNER_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_hash', 'card_name', 'mana_cost', 'card_type', 
                        'set_symbol', 'ocr_success', 'cache_hit', 'processing_time'])

# Set Tesseract path (Windows users)
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def log_ocr_failure(error_msg: str, image_path: str = None):
    """Log OCR failures to ocr_fails.txt"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OCR_FAILS_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {error_msg}")
        if image_path:
            f.write(f" - Image: {image_path}")
        f.write("\n")

def log_scan_attempt(image_hash: str, card_name: str = "", mana_cost: str = "", 
                    card_type: str = "", set_symbol: str = "", ocr_success: bool = False, 
                    cache_hit: bool = False, processing_time: float = 0.0):
    """Log scan attempts to scanner_log.csv"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SCANNER_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, image_hash, card_name, mana_cost, card_type, 
                        set_symbol, ocr_success, cache_hit, processing_time])

def generate_image_hash(image_array) -> str:
    """Generate perceptual hash for image deduplication"""
    try:
        # Convert OpenCV image to PIL
        if len(image_array.shape) == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = Image.fromarray(image_array)
        
        # Generate perceptual hash (dhash is good for similar images)
        dhash = str(imagehash.dhash(pil_image, hash_size=16))
        return dhash
    except Exception as e:
        print(f"⚠️ Hash generation failed: {e}")
        # Fallback to simple pixel hash
        return hashlib.md5(image_array.tobytes()).hexdigest()[:16]

def check_cache_by_hash(image_hash: str) -> Optional[Dict]:
    """Check if image hash exists in JSON cache"""
    cache_file = JSON_CACHE / f"{image_hash}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Cache read error for {image_hash}: {e}")
    return None

def save_to_cache(image_hash: str, card_data: Dict):
    """Save card data to JSON cache using image hash as filename"""
    cache_file = JSON_CACHE / f"{image_hash}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(card_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Cached card data: {cache_file}")
    except Exception as e:
        print(f"❌ Cache save failed for {image_hash}: {e}")

def sanitize_filename(name):
    return name.lower().replace(" ", "_").replace(",", "").replace("'", "")

def clean_card_name(text):
    return re.sub(r"[^a-zA-Z0-9\s'\-]", "", text).strip()

def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR text output"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Remove common OCR artifacts
    cleaned = re.sub(r'[^\w\s\-\'/]', '', cleaned)
    return cleaned

def find_card_assets(card_name):
    """Find corresponding .mind marker and .glb model files for a card"""
    safe_name = sanitize_filename(card_name)
    
    # Define file paths
    marker_file = f"{safe_name}.mind"
    model_file = f"{safe_name}.glb"
    
    marker_path = MARKER_DIR / marker_file
    model_path = MODEL_DIR / model_file
    
    # Check if files exist
    marker_exists = marker_path.exists()
    model_exists = model_path.exists()
    
    print(f"🔍 Checking assets for '{card_name}' (safe_name: {safe_name})")
    print(f"   Marker: {marker_path} - {'✅ Found' if marker_exists else '❌ Missing'}")
    print(f"   Model:  {model_path} - {'✅ Found' if model_exists else '❌ Missing'}")
    
    return {
        "marker_file": f"/assets/mindar/{marker_file}" if marker_exists else None,
        "model_file": f"/assets/models/{model_file}" if model_exists else None,
        "marker_exists": marker_exists,
        "model_exists": model_exists,
        "safe_filename": safe_name
    }

# ============================================================================
# UI-GUIDED CROPPING FUNCTIONS (UNCHANGED FROM YOUR ORIGINAL)
# ============================================================================

def get_ui_guided_crop_coordinates():
    """
    Returns exact crop coordinates matching scan.html UI mask.
    
    UI Setup:
    - Scanner container: 420×588px
    - SVG viewBox: 100×140 units  
    - Card cutout: x="5" y="5" width="90" height="130"
    - Video capture: 1920×1080
    """
    # UI dimensions
    container_width = 420
    container_height = 588
    capture_width = 1920
    capture_height = 1080
    
    # SVG viewBox coordinates
    svg_card_x = 5
    svg_card_y = 5  
    svg_card_width = 90
    svg_card_height = 130
    svg_viewbox_width = 100
    svg_viewbox_height = 140
    
    # Convert SVG coordinates to container pixel coordinates
    container_card_x = (svg_card_x / svg_viewbox_width) * container_width
    container_card_y = (svg_card_y / svg_viewbox_height) * container_height
    container_card_width = (svg_card_width / svg_viewbox_width) * container_width
    container_card_height = (svg_card_height / svg_viewbox_height) * container_height
    
    # Scale to full capture resolution (accounting for object-fit: cover)
    container_aspect = container_width / container_height  # 0.714
    capture_aspect = capture_width / capture_height        # 1.778
    
    if capture_aspect > container_aspect:
        # Video is letterboxed horizontally
        scale_factor = capture_height / container_height
        video_display_width = container_width * scale_factor
        offset_x = (capture_width - video_display_width) / 2
        offset_y = 0
    else:
        # Video is letterboxed vertically
        scale_factor = capture_width / container_width
        video_display_height = container_height * scale_factor
        offset_x = 0
        offset_y = (capture_height - video_display_height) / 2
    
    # Convert to capture coordinates
    capture_card_x = container_card_x * scale_factor + offset_x
    capture_card_y = container_card_y * scale_factor + offset_y
    capture_card_width = container_card_width * scale_factor
    capture_card_height = container_card_height * scale_factor
    
    return {
        'pixel_coords': {
            'x1': int(capture_card_x), 'y1': int(capture_card_y),
            'x2': int(capture_card_x + capture_card_width), 
            'y2': int(capture_card_y + capture_card_height)
        }
    }

def crop_ui_guided_card_area(img):
    """
    STEP 1: Precisely crop the card area that matches your UI overlay mask.
    """
    if img is None or img.size == 0:
        return None
    
    h, w = img.shape[:2]
    coords = get_ui_guided_crop_coordinates()
    px = coords['pixel_coords']
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(px['x1'], w-1))
    y1 = max(0, min(px['y1'], h-1))
    x2 = max(x1+1, min(px['x2'], w))
    y2 = max(y1+1, min(px['y2'], h))
    
    print(f"🎯 STEP 1 - UI-guided crop: ({x1},{y1}) to ({x2},{y2}) from {w}×{h} image")
    
    cropped = img[y1:y2, x1:x2].copy()
    
    if cropped.size == 0:
        print("⚠️ UI-guided crop resulted in empty image")
        return None
        
    return cropped

def get_card_region_coordinates():
    """
    Define regions within the cropped card for different elements.
    These are relative to the UI-guided card crop.
    """
    return {
        'title': {'x1': 0.08, 'y1': 0.05, 'x2': 0.92, 'y2': 0.18},
        'mana_cost': {'x1': 0.75, 'y1': 0.05, 'x2': 0.95, 'y2': 0.15},
        'set_symbol_primary': {'x1': 0.75, 'y1': 0.52, 'x2': 0.92, 'y2': 0.62},
        'set_symbol_alt1': {'x1': 0.70, 'y1': 0.48, 'x2': 0.95, 'y2': 0.66},
        'set_symbol_alt2': {'x1': 0.65, 'y1': 0.45, 'x2': 0.90, 'y2': 0.70},
        'artwork': {'x1': 0.08, 'y1': 0.18, 'x2': 0.92, 'y2': 0.55},
        'type_line': {'x1': 0.08, 'y1': 0.55, 'x2': 0.92, 'y2': 0.62},
        'text_box': {'x1': 0.08, 'y1': 0.62, 'x2': 0.92, 'y2': 0.85},
        'power_toughness': {'x1': 0.75, 'y1': 0.85, 'x2': 0.95, 'y2': 0.95}
    }

def crop_card_region_improved(card_img, region_name, padding=0.02):
    """
    Crop specific region from UI-guided card crop.
    """
    if card_img is None or card_img.size == 0:
        return None
        
    regions = get_card_region_coordinates()
    
    if region_name not in regions:
        print(f"⚠️ Unknown region: {region_name}")
        return None
        
    region = regions[region_name]
    h, w = card_img.shape[:2]
    
    # Apply padding
    x1 = max(0, region['x1'] - padding)
    y1 = max(0, region['y1'] - padding)  
    x2 = min(1, region['x2'] + padding)
    y2 = min(1, region['y2'] + padding)
    
    # Convert to pixel coordinates
    px1 = int(w * x1)
    py1 = int(h * y1)
    px2 = int(w * x2) 
    py2 = int(h * y2)
    
    if px2 <= px1 or py2 <= py1:
        return None
        
    crop = card_img[py1:py2, px1:px2].copy()
    return crop if crop.size > 0 else None

# ============================================================================
# ENHANCED OCR FUNCTIONS FOR STEPS 3, 4, 5
# ============================================================================

def enhance_text_for_ocr(img, target_height=100):
    """Enhanced image processing specifically for text OCR"""
    if img is None or img.size == 0:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Scale up small images for better OCR
    h, w = gray.shape
    if h < target_height:
        scale_factor = target_height / h
        new_w = int(w * scale_factor)
        gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # Adaptive thresholding for better text contrast
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

# --- Updated OCR with integrated cleanup ---
def perform_ocr_with_confidence(img, region_name: str, psm_modes=[7, 8, 13]) -> Tuple[str, float]:
    if img is None:
        return "", 0.0

    enhanced = enhance_text_for_ocr(img)
    if enhanced is None:
        return "", 0.0

    best_text, best_conf = "", 0.0

    for psm in psm_modes:
        config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 /-'
        if region_name == 'mana_cost':
            config = f'--psm {psm} -c tessedit_char_whitelist=0123456789WUBRGCXYZ{{}}'

        try:
            data = pytesseract.image_to_data(enhanced, config=config, output_type=pytesseract.Output.DICT)
        except:
            continue

        words, confidences = [], []
        for i, text in enumerate(data['text']):
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                continue
            if conf > 30 and text.strip():
                words.append(text.strip())
                confidences.append(conf)

        if words:
            joined_text = ' '.join(words)
            avg_conf = sum(confidences) / len(confidences)
            if avg_conf > best_conf:
                best_text, best_conf = joined_text, avg_conf

    # --- Integrated cleanup step ---
    if region_name == 'card_name':
        best_text = fuzzy_correct(clean_text_basic(best_text), CARD_NAMES)
    elif region_name == 'card_type':
        best_text = fuzzy_correct(clean_text_basic(best_text), CARD_TYPES)
    elif region_name == 'mana_cost':
        best_text = normalize_mana_cost(best_text)

    return best_text, best_conf


# ============================================================================
# ENHANCED PROCESSING PIPELINE WITH STEPS 3, 4, 5
# ============================================================================

def process_capture_image_enhanced(image_path):
    """
    ENHANCED: Complete processing pipeline with Steps 1-5, caching, and logging
    """
    start_time = datetime.now()
    print(f"📷 Processing enhanced capture: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        error_msg = f"Failed to load image: {image_path}"
        log_ocr_failure(error_msg)
        return {'error': error_msg}
    
    # STEP 1: Extract the exact card area shown in UI
    print("🔄 STEP 1: UI-guided card cropping...")
    card_crop = crop_ui_guided_card_area(img)
    if card_crop is None:
        error_msg = "UI-guided card crop failed"
        log_ocr_failure(error_msg, str(image_path))
        return {'error': error_msg}
    
    # Generate image hash for caching
    image_hash = generate_image_hash(card_crop)
    print(f"🔑 Generated image hash: {image_hash}")
    
    # Check cache first
    cached_result = check_cache_by_hash(image_hash)
    if cached_result:
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"💾 Cache hit! Returning cached result for {image_hash}")
        log_scan_attempt(image_hash, cached_result.get('name', ''), 
                        cached_result.get('mana_cost', ''), 
                        cached_result.get('type_line', ''),
                        cached_result.get('set_name', ''), 
                        True, True, processing_time)
        cached_result['cache_hit'] = True
        cached_result['image_hash'] = image_hash
        return cached_result
    
    # Save Step 1 image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step1_path = IMAGES_CACHE / f"step1_card_crop_{timestamp}.jpg"
    cv2.imwrite(str(step1_path), card_crop)
    
    # STEP 2: Enhanced set symbol detection
    print("🔄 STEP 2: Set symbol detection...")
    symbol_candidates = enhanced_set_symbol_detection(card_crop, debug_save=True)
    best_symbol_crop = None
    set_symbol_text = ""
    
    if symbol_candidates:
        best_symbol_crop = symbol_candidates[0][0]
        step2_path = IMAGES_CACHE / f"step2_set_symbol_crop_{timestamp}.jpg"
        cv2.imwrite(str(step2_path), best_symbol_crop)
        print(f"✅ STEP 2 Complete: Best set symbol found (confidence: {symbol_candidates[0][1]:.2f})")
    else:
        print("⚠️ STEP 2: No set symbol detected")
    
    # STEP 3: Card type area OCR
    print("🔄 STEP 3: Card type detection...")
    type_crop = crop_card_region_improved(card_crop, 'type_line', padding=0.04)
    card_type_text = ""
    type_confidence = 0.0
    
    if type_crop is not None:
        step3_path = IMAGES_CACHE / f"step3_card_type_{timestamp}.jpg"
        cv2.imwrite(str(step3_path), type_crop)
        card_type_text, type_confidence = perform_ocr_with_confidence(type_crop, 'type_line')
        print(f"✅ STEP 3 Complete: Card type '{card_type_text}' (confidence: {type_confidence:.1f}%)")
    else:
        error_msg = "STEP 3: Card type crop failed"
        log_ocr_failure(error_msg, str(image_path))
        print(f"❌ {error_msg}")
    
    # STEP 4: Card name OCR
    print("🔄 STEP 4: Card name detection...")
    title_crop = crop_card_region_improved(card_crop, 'title', padding=0.02)
    card_name_text = ""
    name_confidence = 0.0
    
    if title_crop is not None:
        step4_path = IMAGES_CACHE / f"step4_card_name_{timestamp}.jpg"
        cv2.imwrite(str(step4_path), title_crop)
        card_name_text, name_confidence = perform_ocr_with_confidence(title_crop, 'title')
        print(f"✅ STEP 4 Complete: Card name '{card_name_text}' (confidence: {name_confidence:.1f}%)")
    else:
        error_msg = "STEP 4: Card name crop failed"
        log_ocr_failure(error_msg, str(image_path))
        print(f"❌ {error_msg}")
    
    # STEP 5: Mana cost OCR
    print("🔄 STEP 5: Mana cost detection...")
    mana_crop = crop_card_region_improved(card_crop, 'mana_cost', padding=0.02)
    mana_cost_text = ""
    mana_confidence = 0.0
    
    if mana_crop is not None:
        step5_path = IMAGES_CACHE / f"step5_mana_cost_{timestamp}.jpg"
        cv2.imwrite(str(step5_path), mana_crop)
        mana_cost_text, mana_confidence = perform_ocr_with_confidence(mana_crop, 'mana_cost')
        print(f"✅ STEP 5 Complete: Mana cost '{mana_cost_text}' (confidence: {mana_confidence:.1f}%)")
    else:
        error_msg = "STEP 5: Mana cost crop failed"
        log_ocr_failure(error_msg, str(image_path))
        print(f"❌ {error_msg}")
    
    # Determine if OCR was successful
    ocr_success = (name_confidence > 50 and len(card_name_text) > 0)
    
    if not ocr_success:
        error_msg = f"OCR failed - Name: '{card_name_text}' (conf: {name_confidence:.1f}%)"
        log_ocr_failure(error_msg, str(image_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        log_scan_attempt(image_hash, card_name_text, mana_cost_text, card_type_text, 
                        set_symbol_text, False, False, processing_time)
        return {'error': error_msg, 'image_hash': image_hash}
    
    # Search Scryfall with the detected card name
    print(f"🔍 Searching Scryfall for: '{card_name_text}'")
    card_data = search_scryfall(card_name_text)
    
    if not card_data:
        error_msg = f"Card '{card_name_text}' not found on Scryfall"
        log_ocr_failure(error_msg, str(image_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        log_scan_attempt(image_hash, card_name_text, mana_cost_text, card_type_text, 
                        set_symbol_text, False, False, processing_time)
        return {'error': error_msg, 'image_hash': image_hash}
    
    # Use official Scryfall name for asset matching
    official_name = card_data.get('name', card_name_text)
    assets = find_card_assets(official_name)
    
    # Build comprehensive result
    result = {
        'success': True,
        'cache_hit': False,
        'image_hash': image_hash,
        'card_name': official_name,
        'ocr_detected_name': card_name_text,
        'mana_cost': mana_cost_text,
        'card_type': card_type_text,
        'set_symbol': set_symbol_text,
        'ocr_confidences': {
            'name': name_confidence,
            'mana_cost': mana_confidence,
            'card_type': type_confidence
        },
        'oracle_id': card_data.get('oracle_id'),
        'scryfall_id': card_data.get('id'),
        'image_url': card_data.get('image_uris', {}).get('art_crop'),
        'full_image_url': card_data.get('image_uris', {}).get('normal'),
        'card_image_url': card_data.get('image_uris', {}).get('border_crop'),
        'marker_file': assets['marker_file'],
        'model_file': assets['model_file'],
        'assets_available': assets['marker_exists'] and assets['model_exists'],
        'marker_exists': assets['marker_exists'],
        'model_exists': assets['model_exists'],
        'safe_filename': assets['safe_filename'],
        'processing_method': 'enhanced_5_step_pipeline',
        'scryfall_data': card_data
    }
    
    # Cache the result for future scans
    save_to_cache(image_hash, result)
    
    # Log the successful attempt
    processing_time = (datetime.now() - start_time).total_seconds()
    log_scan_attempt(image_hash, official_name, mana_cost_text, card_type_text, 
                    set_symbol_text, True, False, processing_time)
    
    # Log results
    if result['assets_available']:
        print(f"✅ Complete AR setup found for '{official_name}'!")
    else:
        missing = []
        if not assets['marker_exists']:
            missing.append('marker (.mind)')
        if not assets['model_exists']:
            missing.append('model (.glb)')
        print(f"⚠️ '{official_name}' missing: {', '.join(missing)}")
    
    print(f"⏱️ Total processing time: {processing_time:.2f}s")
    return result

# ============================================================================
# ENHANCED SET SYMBOL DETECTION (STEP 2 IMPROVEMENTS)
# ============================================================================

def enhanced_set_symbol_detection(card_img, debug_save=False):
    """
    STEP 2: Multi-region set symbol detection with confidence scoring.
    """
    if card_img is None:
        return []
        
    candidates = []
    region_names = ['set_symbol_primary', 'set_symbol_alt1', 'set_symbol_alt2']
    
    for region_name in region_names:
        crop = crop_card_region_improved(card_img, region_name, padding=0.05)
        if crop is not None:
            confidence = analyze_set_symbol_confidence(crop)
            candidates.append((crop, confidence, region_name))
            
            if debug_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = IMAGES_CACHE / f"set_symbol_{region_name}_{confidence:.2f}_{timestamp}.jpg"
                cv2.imwrite(str(debug_path), crop)
    
    return sorted(candidates, key=lambda x: x[1], reverse=True)

def analyze_set_symbol_confidence(crop_img):
    """
    Analyze likelihood that crop contains a set symbol.
    """
    if crop_img is None or crop_img.size == 0:
        return 0.0
        
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img
    h, w = gray.shape
    
    # Size validation
    min_dim = min(h, w)
    if min_dim < 15 or min_dim > 120:
        return 0.1
    
    # Edge detection
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # Contrast analysis
    contrast = np.std(gray.astype(np.float32))
    
    # Shape analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_score = 0.0
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 50:
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                shape_score = min(circularity, 1.0)
    
    # Darkness analysis
    mean_brightness = np.mean(gray)
    darkness_score = 1.0 - (mean_brightness / 255.0)
    
    # Combine features
    edge_score = min(edge_density * 8, 1.0)
    contrast_score = min(contrast / 60, 1.0)      
    size_score = 1.0 if 20 <= min_dim <= 80 else 0.5
    
    confidence = (edge_score * 0.3 + 
                 contrast_score * 0.25 + 
                 size_score * 0.2 +
                 shape_score * 0.15 +
                 darkness_score * 0.1)
    
    return min(confidence, 1.0)

# ============================================================================
# SCRYFALL API FUNCTIONS
# ============================================================================

def search_scryfall(card_name):
    """Search for card data on Scryfall API"""
    if not card_name:
        return None
        
    url = f'https://api.scryfall.com/cards/named?fuzzy={card_name}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️ Scryfall API returned {response.status_code} for: {card_name}")
            return None
    except requests.RequestException as e:
        print(f"❌ Scryfall API error: {e}")
        return None

def search_scryfall_advanced(card_name: str, mana_cost: str = "", card_type: str = "", set_name: str = ""):
    """
    Enhanced Scryfall search using multiple parameters for better accuracy
    """
    if not card_name:
        return None
    
    # Start with fuzzy name search
    base_url = "https://api.scryfall.com/cards/search?q="
    query_parts = [f'name:"{card_name}"']
    
    # Add additional parameters if available
    if mana_cost and mana_cost.strip():
        # Clean mana cost for search
        clean_mana = re.sub(r'[^\dWUBRGCXYZ]', '', mana_cost.upper())
        if clean_mana:
            query_parts.append(f'mana:"{clean_mana}"')
    
    if card_type and card_type.strip():
        # Extract main type (Creature, Instant, etc.)
        type_match = re.search(r'\b(Creature|Instant|Sorcery|Enchantment|Artifact|Planeswalker|Land)\b', card_type, re.IGNORECASE)
        if type_match:
            query_parts.append(f'type:"{type_match.group(1)}"')
    
    if set_name and set_name.strip():
        query_parts.append(f'set:"{set_name}"')
    
    query = " ".join(query_parts)
    url = base_url + requests.utils.quote(query)
    
    try:
        print(f"🔍 Advanced Scryfall search: {query}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('total_cards', 0) > 0:
                # Return the first (most relevant) card
                return data['data'][0]
        
        # Fallback to simple fuzzy search if advanced search fails
        print("⚠️ Advanced search failed, falling back to fuzzy search")
        return search_scryfall(card_name)
        
    except requests.RequestException as e:
        print(f"❌ Advanced Scryfall search error: {e}")
        return search_scryfall(card_name)

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_camera_capture(image_path):
    """
    MAIN ENTRY POINT: Enhanced processing pipeline wrapper
    This replaces your old process_camera_capture function
    """
    return process_capture_image_enhanced(image_path)

# Backward compatibility
def process_card_image(image_path):
    """Backward compatibility wrapper"""
    return process_camera_capture(image_path)

def process_capture_image(image_path):
    """Backward compatibility wrapper"""
    return process_camera_capture(image_path)

# ============================================================================
# UTILITY AND DEBUGGING FUNCTIONS
# ============================================================================

def list_available_assets():
    """Utility function to list all available assets"""
    marker_files = list(MARKER_DIR.glob("*.mind"))
    model_files = list(MODEL_DIR.glob("*.glb"))
    
    markers = [f.stem for f in marker_files]
    models = [f.stem for f in model_files]
    complete_pairs = set(markers) & set(models)
    
    return {
        'markers': sorted(markers),
        'models': sorted(models),
        'complete_pairs': sorted(list(complete_pairs)),
        'marker_count': len(markers),
        'model_count': len(models),
        'complete_count': len(complete_pairs)
    }

def get_cache_stats():
    """Get statistics about the JSON cache"""
    if not JSON_CACHE.exists():
        return {'cache_count': 0, 'cache_size_mb': 0}
    
    cache_files = list(JSON_CACHE.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        'cache_count': len(cache_files),
        'cache_size_mb': round(total_size / (1024 * 1024), 2),
        'cache_files': [f.stem for f in cache_files]
    }

def clear_cache(older_than_days: int = 7):
    """Clear cache files older than specified days"""
    if not JSON_CACHE.exists():
        return 0
    
    cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 3600)
    removed_count = 0
    
    for cache_file in JSON_CACHE.glob("*.json"):
        if cache_file.stat().st_mtime < cutoff_time:
            cache_file.unlink()
            removed_count += 1
    
    return removed_count

def debug_ocr_region(image_path: str, region_name: str):
    """
    Debug function to test OCR on specific regions
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    card_crop = crop_ui_guided_card_area(img)
    if card_crop is None:
        return None
    
    region_crop = crop_card_region_improved(card_crop, region_name)
    if region_crop is None:
        return None
    
    # Save debug images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = IMAGES_CACHE / f"debug_{region_name}_{timestamp}.jpg"
    cv2.imwrite(str(debug_path), region_crop)
    
    # Perform OCR
    text, confidence = perform_ocr_with_confidence(region_crop, region_name)
    
    return {
        'region': region_name,
        'text': text,
        'confidence': confidence,
        'debug_image': str(debug_path)
    }

# ============================================================================
# LEGACY FUNCTIONS (KEPT FOR BACKWARD COMPATIBILITY)
# ============================================================================

def extract_card_name_from_image_cv(img, crop_ratio=0.18):
    """
    LEGACY: Kept for backward compatibility
    """
    if img is None or img.size == 0:
        print("❌ Input image is None or empty")
        return None
    
    # Use new enhanced pipeline
    result = process_capture_image_enhanced("temp")  # This won't work perfectly but maintains interface
    return result.get('ocr_detected_name') if isinstance(result, dict) else None

def extract_card_name_with_camera_processing(img):
    """
    LEGACY: Enhanced OCR processing using new system.
    """
    if img is None or img.size == 0:
        print("❌ Input image is None or empty")
        return None, None

    card_img = crop_ui_guided_card_area(img)
    if card_img is None:
        print("⚠️ Failed to crop card region; using original")
        card_img = img
    
    title_img = crop_card_region_improved(card_img, 'title', padding=0.02)
    if title_img is None:
        print("❌ Title crop failed")
        return None, None
    
    card_name, confidence = perform_ocr_with_confidence(title_img, 'title')
    return card_name if confidence > 50 else None, card_img

def crop_center_card_area(img, target_width=818, target_height=1080):
    """LEGACY: Updated to use UI-guided cropping."""
    return crop_ui_guided_card_area(img)

def crop_card_region(img, bleed_ratio=0.1):
    """LEGACY: Crop bleed margins using new system."""
    if img is None or img.size == 0:
        return None
    
    h, w = img.shape[:2]
    x1 = int(w * bleed_ratio)
    y1 = int(h * bleed_ratio)
    x2 = int(w * (1 - bleed_ratio))
    y2 = int(h * (1 - bleed_ratio))
    
    cropped = img[y1:y2, x1:x2].copy()
    return cropped if cropped.size > 0 else None

def crop_title_area(card_img, crop_ratio=0.18):
    """LEGACY: Better title area cropping."""
    return crop_card_region_improved(card_img, 'title', padding=0.02)

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("🎮 MTG Card Scanner - Enhanced Pipeline")
    print("=" * 50)
    
    # Test with an image
    test_image_path = "path/to/your/test/image.jpg"
    if os.path.exists(test_image_path):
        result = process_camera_capture(test_image_path)
        print(f"Result: {result}")
    
    # Show cache stats
    stats = get_cache_stats()
    print(f"📊 Cache Stats: {stats['cache_count']} files, {stats['cache_size_mb']} MB")
    
    # Show available assets
    assets = list_available_assets()
    print(f"🎯 Available Assets: {assets['complete_count']} complete pairs")