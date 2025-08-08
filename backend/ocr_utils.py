import pytesseract
import requests
import os
import cv2
import re
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple, List

# Define directories
BASE_DIR = Path(__file__).resolve().parent.parent
MARKER_DIR = BASE_DIR / "public" / "assets" / "mindar"
MODEL_DIR = BASE_DIR / "public" / "assets" / "models"
CACHE_DIR = BASE_DIR / "cache" / "images"



# Create directories if they don't exist
MARKER_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set Tesseract path (Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def sanitize_filename(name):
    return name.lower().replace(" ", "_").replace(",", "").replace("'", "")

def clean_card_name(text):
    return re.sub(r"[^a-zA-Z0-9\s'\-]", "", text).strip()

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


# -------------------------
# Step 1: Center crop full card
# -------------------------
def crop_center_card(img, target_w=818, target_h=1080):
    h, w = img.shape[:2]
    x1 = (w - target_w) // 2
    y1 = (h - target_h) // 2
    return img[y1:y1+target_h, x1:x1+target_w]

# -------------------------
# Step 2: Set symbol crop + match
# -------------------------
def crop_set_symbol(card_img):
    h, w = card_img.shape[:2]
    # Approx coords for MTG set symbol in 3:4 crop
    x1 = int(w * 0.72)
    y1 = int(h * 0.42)

def crop_center_card_area(img, target_width=818, target_height=1080):
    """
    Crop the center 818x1080 area (3:4) from the input image (e.g., 1920x1080)
    """
    if img is None or img.size == 0:
        return None
    
    h, w = img.shape[:2]
    if h < target_height or w < target_width:
        print(f"⚠️ Input image is smaller than target crop size: {w}x{h}")
        return None
    
    start_x = (w - target_width) // 2
    start_y = (h - target_height) // 2  # Usually zero if input height = target_height
    
    cropped = img[start_y:start_y+target_height, start_x:start_x+target_width].copy()
    return cropped

def crop_area_from_card(card_img, x_ratio_start, y_ratio_start, x_ratio_end, y_ratio_end):
    """
    Crop a sub-area from the card image given normalized ratios.
    All ratios are floats between 0 and 1 relative to card_img dimensions.
    """
    if card_img is None or card_img.size == 0:
        return None
    h, w = card_img.shape[:2]
    x1 = int(w * x_ratio_start)
    y1 = int(h * y_ratio_start)
    x2 = int(w * x_ratio_end)
    y2 = int(h * y_ratio_end)
    crop = card_img[y1:y2, x1:x2]
    if crop.size == 0:
        print("⚠️ Warning: crop_area_from_card returned empty image.")
        return None
    return crop

def detect_card_contour(img, min_area_ratio=0.1, max_area_ratio=0.9):
    """
    Detect MTG card contour using edge detection and aspect ratio validation
    MTG cards: 63mm × 88mm (2.5" × 3.5"), ratio 0.715
    """
    if img is None or img.size == 0:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection with adjusted parameters for card detection
    edges = cv2.Canny(blurred, 75, 200, apertureSize=3)
    
    # Morphological operations to clean up edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # MTG card aspect ratio (width/height = 63/88 = 0.715)
    target_ratio = 63.0 / 88.0  # 0.715
    ratio_tolerance = 0.2  # Allow 20% deviation
    
    # Filter contours by area and aspect ratio
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Approximate contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes (4 corners)
            if len(approx) == 4:
                # Check aspect ratio
                rect = cv2.boundingRect(approx)
                w, h = rect[2], rect[3]
                
                if h > 0:  # Avoid division by zero
                    aspect_ratio = w / h
                    
                    # Check if aspect ratio matches MTG card (considering both orientations)
                    ratio_match = (abs(aspect_ratio - target_ratio) <= ratio_tolerance or 
                                 abs(aspect_ratio - (1/target_ratio)) <= ratio_tolerance)
                    
                    if ratio_match:
                        # Calculate contour quality score (area vs bounding box)
                        bbox_area = w * h
                        fill_ratio = area / bbox_area if bbox_area > 0 else 0
                        
                        # Prefer contours that fill their bounding box well (more rectangular)
                        if fill_ratio > 0.8:
                            quality_score = area * fill_ratio
                            valid_contours.append((contour, quality_score, aspect_ratio))
    
    if not valid_contours:
        return None
    
    # Return the highest quality contour
    best_contour = max(valid_contours, key=lambda x: x[1])[0]
    return best_contour

def extract_card_from_contour(img, contour):
    """
    Extract and perspective-correct the MTG card from detected contour
    Uses proper MTG card dimensions: 63mm × 88mm (ratio 0.715)
    """
    if contour is None:
        return None
    
    # Get the four corner points
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 4:
        return None
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    points = approx.reshape(4, 2).astype(np.float32)
    
    # Calculate the center point
    center = np.mean(points, axis=0)
    
    # Sort points by angle from center (clockwise from top-left)
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
    
    # Sort by angle and assign corners
    sorted_points = sorted(points, key=angle_from_center)
    
    # Find the correct orientation based on aspect ratio
    # Calculate width and height of detected card
    top_width = np.linalg.norm(sorted_points[1] - sorted_points[0])
    side_height = np.linalg.norm(sorted_points[3] - sorted_points[0])
    
    detected_ratio = top_width / side_height if side_height > 0 else 1
    mtg_ratio = 63.0 / 88.0  # 0.715
    
    # Determine if card is in portrait (normal) or landscape orientation
    if abs(detected_ratio - mtg_ratio) < abs(detected_ratio - (1/mtg_ratio)):
        # Card is in landscape orientation (wider than tall)
        card_width = 315  # 63mm * 5 pixels/mm
        card_height = 440  # 88mm * 5 pixels/mm
    else:
        # Card is in portrait orientation (taller than wide) 
        card_width = 440  # 88mm * 5 pixels/mm  
        card_height = 315  # 63mm * 5 pixels/mm
    
    # Re-order points for perspective transform
    # We want: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        # Sort by y-coordinate (top vs bottom)
        sorted_y = sorted(pts, key=lambda p: p[1])
        
        # Top two points
        top_pts = sorted_y[:2]
        # Bottom two points  
        bottom_pts = sorted_y[2:]
        
        # Sort top points by x-coordinate (left vs right)
        top_pts = sorted(top_pts, key=lambda p: p[0])
        # Sort bottom points by x-coordinate (left vs right)
        bottom_pts = sorted(bottom_pts, key=lambda p: p[0])
        
        return np.array([
            top_pts[0],     # top-left
            top_pts[1],     # top-right
            bottom_pts[1],  # bottom-right
            bottom_pts[0]   # bottom-left
        ], dtype=np.float32)
    
    src_points = order_points(points)
    
    # Define destination points for perspective correction
    dst_points = np.float32([
        [0, 0],                           # top-left
        [card_width, 0],                  # top-right
        [card_width, card_height],        # bottom-right
        [0, card_height]                  # bottom-left
    ])
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective correction
    corrected = cv2.warpPerspective(img, matrix, (card_width, card_height))
    
    # Save debug image showing detected corners
    debug_img = img.copy()
    for i, point in enumerate(src_points):
        cv2.circle(debug_img, tuple(point.astype(int)), 10, (0, 255, 0), -1)
        cv2.putText(debug_img, str(i), tuple(point.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(str(CACHE_DIR / "card_detection_debug.jpg"), debug_img)
    
    return corrected

def crop_card_region(img, bleed_ratio=0.1):
    """
    Crop the card region to remove bleed margins around edges.
    """
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    x1 = int(w * bleed_ratio)
    y1 = int(h * bleed_ratio)
    x2 = int(w * (1 - bleed_ratio))
    y2 = int(h * (1 - bleed_ratio))
    cropped = img[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        print("⚠️ Warning: crop_card_region returned empty image.")
        return None
    return cropped

def crop_title_area(card_img, crop_ratio=0.15):
    """
    Crop the top portion of the card (title area).
    """
    if card_img is None or card_img.size == 0:
        return None
    h, w = card_img.shape[:2]
    crop = card_img[0:int(h * crop_ratio), 0:w].copy()
    if crop.size == 0:
        print("⚠️ Warning: crop_title_area returned empty image.")
        return None
    return crop

def extract_card_name_from_image_cv(img, crop_ratio=0.15):
    """
    Simple OCR extraction:
    - Crop bleed margins from the card
    - Crop to title area inside card
    - Convert to grayscale and denoise
    - Run pytesseract OCR
    """
    if img is None or img.size == 0:
        print("❌ Input image is None or empty")
        return None
    
    # Crop bleed margin area
    card_img = crop_card_region(img, bleed_ratio=0.1)
    if card_img is None:
        card_img = img  # fallback
    
    # Crop title area inside card image
    title_img = crop_title_area(card_img, crop_ratio)
    if title_img is None:
        title_img = card_img  # fallback

    gray = cv2.cvtColor(title_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # Save debug images
    cv2.imwrite(str(CACHE_DIR / "ocr_input.jpg"), img)
    cv2.imwrite(str(CACHE_DIR / "ocr_card_crop.jpg"), card_img)
    cv2.imwrite(str(CACHE_DIR / "ocr_title_crop.jpg"), gray)

    text = pytesseract.image_to_string(gray, config='--psm 7')
    cleaned = clean_card_name(text.strip())

    print(f"🔍 OCR result: {cleaned}")
    return cleaned if cleaned else None

def enhance_text_for_ocr(img):
    """
    Apply image processing techniques to improve OCR accuracy
    """
    if img is None or img.size == 0:
        return None
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply various enhancement techniques
    # 1. Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # 2. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 3. Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return morph

def extract_card_name_with_camera_processing(img):
    """
    Updated to:
    - Crop the center 818x1080 card area first
    - Then detect card contour inside that crop
    - Then extract and perspective correct the card inside that crop
    - Then crop title and OCR
    """
    if img is None or img.size == 0:
        print("❌ Input image is None or empty")
        return None, None

    original_img = img.copy()
    
    # Step 1: Crop center 818x1080 from original capture
    card_region_img = crop_center_card_area(img, 818, 1080)
    if card_region_img is None:
        print("⚠️ Failed to crop center card region; using original image")
        card_region_img = img
    
    # Save debug
    cv2.imwrite(str(CACHE_DIR / "cropped_card_region.jpg"), card_region_img)
    
    # Step 2: Detect card contour within this cropped card region
    card_contour = detect_card_contour(card_region_img)
    
    if card_contour is not None:
        # Step 3: Extract and correct perspective from cropped card region
        card_img = extract_card_from_contour(card_region_img, card_contour)
        
        if card_img is not None:
            print("✅ Card detected and extracted automatically")
            
            # Save debug image of extracted card
            cv2.imwrite(str(CACHE_DIR / "detected_card.jpg"), card_img)
        else:
            print("⚠️ Card detection failed, using card region crop")
            card_img = card_region_img
    else:
        print("⚠️ No card contour detected, using card region crop")
        card_img = card_region_img
    
    # Step 4: Crop to title area from extracted card image
    title_img = crop_title_area(card_img)
    if title_img is None:
        print("❌ Title crop failed")
        return None, None
    
    # Step 5: Enhance image for OCR
    enhanced_title = enhance_text_for_ocr(title_img)
    if enhanced_title is None:
        print("❌ Image enhancement failed")
        return None, None
    
    # Save debug images
    cv2.imwrite(str(CACHE_DIR / "ocr_input.jpg"), original_img)
    if card_contour is not None and card_img is not None:
        cv2.imwrite(str(CACHE_DIR / "extracted_card.jpg"), card_img)
    cv2.imwrite(str(CACHE_DIR / "title_crop.jpg"), title_img)
    cv2.imwrite(str(CACHE_DIR / "enhanced_title.jpg"), enhanced_title)
    
    # Step 6: Run OCR with multiple configurations
    ocr_configs = [
        '--psm 7',  # Single text line
        '--psm 8',  # Single word
        '--psm 13', # Raw line. Treat image as single text line
    ]
    
    best_result = None
    best_confidence = 0
    
    for config in ocr_configs:
        try:
            data = pytesseract.image_to_data(enhanced_title, config=config, output_type=pytesseract.Output.DICT)
            words = []
            confidences = []
            for i in range(len(data['text'])):
                try:
                    conf = float(data['conf'][i])
                except ValueError:
                    continue
                if conf > 30:
                    word = data['text'][i].strip()
                    if word:
                        words.append(word)
                        confidences.append(conf)
            if words and confidences:
                text = ' '.join(words)
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence > best_confidence:
                    best_result = text
                    best_confidence = avg_confidence
        except Exception as e:
            print(f"⚠️ OCR config {config} failed: {e}")
            continue
    
    if best_result:
        cleaned = clean_card_name(best_result)
        print(f"🔍 Best OCR result: '{cleaned}' (confidence: {best_confidence:.1f}%)")
        return cleaned, card_img
    
    print("❌ No valid OCR results")
    return None, card_img

def search_scryfall(card_name):
    """Search for card data on Scryfall API with better error handling"""
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

def process_camera_capture(image_path):
    """
    Enhanced processing pipeline optimized for camera capture
    Includes automatic card detection, perspective correction, and asset matching
    """
    print(f"📷 Processing camera capture: {image_path}")
    
    # Load and validate image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to read image at path: {image_path}")
        return {'error': f'Image not found or unreadable at {image_path}'}

    # Enhanced card name extraction with camera processing
    card_name, extracted_card = extract_card_name_with_camera_processing(img)
    
    if not card_name:
        return {'error': 'No card name detected via OCR'}

    # Search Scryfall for card data
    card_data = search_scryfall(card_name)
    if not card_data:
        return {'error': f'Card "{card_name}" not found on Scryfall'}

    # Use official Scryfall name for asset matching
    official_name = card_data.get('name', card_name)
    
    # Find corresponding asset files
    assets = find_card_assets(official_name)
    
    # Build comprehensive response
    result = {
        'success': True,
        'card_name': official_name,
        'ocr_detected_name': card_name,
        'oracle_id': card_data.get('oracle_id'),
        'scryfall_id': card_data.get('id'),
        
        # Image URLs from Scryfall
        'image_url': card_data.get('image_uris', {}).get('art_crop'),
        'full_image_url': card_data.get('image_uris', {}).get('normal'),
        'card_image_url': card_data.get('image_uris', {}).get('border_crop'),
        
        # Asset file paths for AR
        'marker_file': assets['marker_file'],
        'model_file': assets['model_file'],
        
        # Asset availability
        'assets_available': assets['marker_exists'] and assets['model_exists'],
        'marker_exists': assets['marker_exists'],
        'model_exists': assets['model_exists'],
        'safe_filename': assets['safe_filename'],
        
        # Card detection info
        'card_detected': extracted_card is not None,
        'processing_method': 'camera_optimized',
        
        # Full Scryfall data
        'scryfall_data': card_data
    }
    
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
    
    return result

# Keep backward compatibility
def process_card_image(image_path):
    """Backward compatibility wrapper"""
    return process_camera_capture(image_path)

def list_available_assets():
    """Utility function to list all available assets"""
    marker_files = list(MARKER_DIR.glob("*.mind"))
    model_files = list(MODEL_DIR.glob("*.glb"))
    
    markers = [f.stem for f in marker_files]
    models = [f.stem for f in model_files]
    
    # Find complete pairs
    complete_pairs = set(markers) & set(models)
    
    return {
        'markers': sorted(markers),
        'models': sorted(models),
        'complete_pairs': sorted(list(complete_pairs)),
        'marker_count': len(markers),
        'model_count': len(models),
        'complete_count': len(complete_pairs)
    }