from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import json
import cv2
import numpy as np
import base64
from typing import List, Optional
import ocr_utils
import asyncio
from datetime import datetime


app = FastAPI(
    title="MTG Card Scanner Camera API",
    description="Real-time Magic: The Gathering card scanning with camera support and AR integration",
    version="3.0.0"  # Updated version for enhanced features
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for camera configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 2560  # Align with frontend requested resolution
CAMERA_HEIGHT = 1440
CAMERA_FPS = 30
MAX_WEBSOCKET_CLIENTS = 10  # Max concurrent websocket connections allowed

# Directory setup
BASE_DIR = Path(__file__).resolve().parent.parent  # Go two levels up from backend/main.py
WEB_DIR = BASE_DIR / "web"
IMAGE_CACHE_DIR = Path("/tmp/cache/images")
CAPTURE_DIR = Path("/tmp/cache/captures")
JSON_CACHE_DIR = Path("/tmp/cache/json_cache")

# Create all required directories
for directory in [IMAGE_CACHE_DIR, CAPTURE_DIR, JSON_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

class CameraManager:
    def __init__(self):
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_active: bool = False
    
    def start_camera(self, camera_index: int = CAMERA_INDEX) -> bool:
        """Initialize camera capture with configured resolution and fps"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_index}")
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            self.is_active = True
            print(f"✅ Camera {camera_index} initialized successfully with {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
            return True
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera"""
        if not self.is_active or self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def release_camera(self) -> None:
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.is_active = False
        print("📷 Camera released")

# Global camera manager
camera_manager = CameraManager()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up camera on shutdown"""
    camera_manager.release_camera()

@app.post("/scan/")
async def scan_card(file: UploadFile = File(...)):
    """
    Enhanced: Scan a Magic: The Gathering card image using the new 5-step pipeline
    Now includes caching, improved OCR, and comprehensive logging
    """
    # Validate file type early
    if not file.content_type.startswith('image/'):
        await file.close()
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded capture to cache
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = f"upload_raw_{timestamp}_{file.filename}"
    raw_path = IMAGE_CACHE_DIR / raw_filename
    
    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        await file.close()
    except Exception as e:
        await file.close()
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    print(f"📷 Processing uploaded file: {raw_path}")

    # Use the enhanced processing pipeline (Steps 1-5 with caching)
    try:
        result = ocr_utils.process_camera_capture(str(raw_path))
        
        # Handle error results
        if isinstance(result, dict) and 'error' in result:
            raise HTTPException(status_code=422, detail=result['error'])
        
        # Add processing metadata
        result['processing_timestamp'] = timestamp
        result['source'] = 'file_upload'
        result['raw_filename'] = raw_filename
        
        # Log successful processing
        print(f"✅ Upload processing complete:")
        print(f"   Card: {result.get('card_name', 'Unknown')}")
        print(f"   Cache Hit: {result.get('cache_hit', False)}")
        print(f"   Assets Available: {result.get('assets_available', False)}")
        
        return JSONResponse(content=result)

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"❌ Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced processing error: {str(e)}")

@app.post("/camera/capture/")
async def capture_from_camera():
    """
    Enhanced: Capture and process a card image directly from device camera using new pipeline
    """
    if not camera_manager.is_active:
        if not camera_manager.start_camera():
            raise HTTPException(status_code=500, detail="Camera initialization failed")
    
    # Capture frame
    frame = camera_manager.capture_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from camera")
    
    # Save captured frame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"camera_capture_{timestamp}.jpg"
    file_path = CAPTURE_DIR / filename
    
    cv2.imwrite(str(file_path), frame)
    
    try:
        # Use the enhanced processing pipeline
        result = ocr_utils.process_camera_capture(str(file_path))
        
        # Handle error results
        if isinstance(result, dict) and 'error' in result:
            raise HTTPException(status_code=422, detail=result['error'])
        
        # Add processing metadata
        result['processing_timestamp'] = timestamp
        result['source'] = 'camera_capture'
        result['captured_image'] = f"/captures/{filename}"
        
        # Log successful processing
        print(f"✅ Camera capture processing complete:")
        print(f"   Card: {result.get('card_name', 'Unknown')}")
        print(f"   Cache Hit: {result.get('cache_hit', False)}")
        print(f"   OCR Confidences: {result.get('ocr_confidences', {})}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Clean up failed capture
        file_path.unlink(missing_ok=True)
        print(f"❌ Camera processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced camera processing error: {str(e)}")

@app.websocket("/ws/camera-stream/")
async def camera_stream_websocket(websocket: WebSocket):
    """
    Enhanced: WebSocket endpoint with improved card detection using new pipeline
    """
    if len(active_connections) >= MAX_WEBSOCKET_CLIENTS:
        await websocket.close(code=1008, reason="Max connections reached")
        return

    await websocket.accept()
    active_connections.append(websocket)
    
    print("📱 Enhanced camera stream WebSocket connected")
    
    try:
        if not camera_manager.is_active:
            if not camera_manager.start_camera():
                await websocket.send_json({
                    "error": "Camera initialization failed",
                    "type": "camera_error"
                })
                return
        
        while True:
            try:
                # Capture frame
                frame = camera_manager.capture_frame()
                if frame is None:
                    await websocket.send_json({
                        "error": "Failed to capture frame",
                        "type": "capture_error"
                    })
                    await asyncio.sleep(0.1)
                    continue
                
                # Encode frame to base64 for streaming
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame to client
                await websocket.send_json({
                    "type": "frame",
                    "image": frame_base64,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check for messages from client (e.g., capture request)
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=0.033)  # ~30 FPS
                    
                    if message.get("action") == "capture":
                        # Process current frame using enhanced pipeline
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"stream_capture_{timestamp}.jpg"
                        file_path = CAPTURE_DIR / filename
                        
                        cv2.imwrite(str(file_path), frame)
                        
                        try:
                            result = ocr_utils.process_camera_capture(str(file_path))
                            
                            # Handle successful results
                            if isinstance(result, dict) and result.get('success'):
                                result['processing_timestamp'] = timestamp
                                result['source'] = 'camera_stream'
                                result['captured_image'] = f"/captures/{filename}"
                                
                                await websocket.send_json({
                                    "type": "scan_result",
                                    **result
                                })
                                
                                print(f"📱 Stream scan success: {result.get('card_name', 'Unknown')}")
                            else:
                                # Handle error results
                                error_msg = result.get('error', 'Unknown processing error') if isinstance(result, dict) else 'Processing failed'
                                await websocket.send_json({
                                    "type": "scan_error",
                                    "error": error_msg
                                })
                            
                        except Exception as e:
                            await websocket.send_json({
                                "type": "scan_error",
                                "error": f"Enhanced processing failed: {str(e)}"
                            })
                
                except asyncio.TimeoutError:
                    # No message from client, continue streaming
                    pass
                    
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "stream_error",
                    "error": str(e)
                })
                break
                
    except WebSocketDisconnect:
        print("📱 Enhanced camera stream WebSocket disconnected")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.post("/camera/start/")
async def start_camera():
    """Start camera for capture operations"""
    if camera_manager.start_camera():
        return {"status": "success", "message": "Camera started successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start camera")

@app.post("/camera/stop/")
async def stop_camera():
    """Stop camera and release resources"""
    camera_manager.release_camera()
    return {"status": "success", "message": "Camera stopped"}

@app.get("/camera/status/")
async def camera_status():
    """Get current camera status"""
    return {
        "active": camera_manager.is_active,
        "connected_clients": len(active_connections)
    }

@app.post("/scan-batch/")
async def scan_multiple_cards(files: List[UploadFile] = File(...)):
    """Enhanced: Scan multiple card images using new pipeline with caching benefits"""
    if len(files) > 10:  # Reasonable limit for batch processing
        for file in files:
            await file.close()
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type - must be image'
            })
            await file.close()
            continue
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{timestamp}_{file.filename}"
        file_path = IMAGE_CACHE_DIR / filename
        
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            await file.close()
            
            # Use enhanced processing pipeline
            result = ocr_utils.process_camera_capture(str(file_path))
            
            # Handle both success and error cases
            if isinstance(result, dict):
                if 'error' in result:
                    results.append({
                        'filename': file.filename,
                        'error': result['error']
                    })
                else:
                    result['filename'] = file.filename
                    result['processing_timestamp'] = timestamp
                    results.append(result)
            else:
                results.append({
                    'filename': file.filename,
                    'error': 'Unexpected result format'
                })
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': f'Processing failed: {str(e)}'
            })
        finally:
            # Clean up temporary file
            if file_path.exists():
                file_path.unlink(missing_ok=True)
    
    return JSONResponse(content={'results': results})

@app.get("/assets/inventory/")
async def get_asset_inventory():
    """Get inventory of available marker and model assets"""
    try:
        inventory = ocr_utils.list_available_assets()
        return JSONResponse(content=inventory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get inventory: {str(e)}")

@app.get("/card/{card_name}/assets/")
async def check_card_assets(card_name: str):
    """Check if a specific card has marker and model assets available"""
    try:
        assets = ocr_utils.find_card_assets(card_name)
        return JSONResponse(content={
            'card_name': card_name,
            **assets
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Asset check failed: {str(e)}")

# NEW ENDPOINTS FOR ENHANCED FEATURES

@app.get("/cache/stats/")
async def get_cache_stats():
    """Get statistics about the enhanced caching system"""
    try:
        stats = ocr_utils.get_cache_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")

@app.post("/cache/clear/")
async def clear_cache(older_than_days: int = 7):
    """Clear cache files older than specified days"""
    try:
        removed_count = ocr_utils.clear_cache(older_than_days)
        return JSONResponse(content={
            "status": "success",
            "removed_files": removed_count,
            "older_than_days": older_than_days
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.post("/debug/ocr/{region_name}/")
async def debug_ocr_region(region_name: str, file: UploadFile = File(...)):
    """Debug OCR performance on specific card regions"""
    if not file.content_type.startswith('image/'):
        await file.close()
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = f"debug_temp_{timestamp}_{file.filename}"
    temp_path = IMAGE_CACHE_DIR / temp_filename
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        await file.close()
        
        # Run debug function
        result = ocr_utils.debug_ocr_region(str(temp_path), region_name)
        
        if result is None:
            raise HTTPException(status_code=422, detail=f"Could not extract region '{region_name}' from image")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug OCR failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

@app.get("/logs/ocr-failures/")
async def get_ocr_failures():
    """Get recent OCR failure logs"""
    try:
        ocr_fails_log = Path("/tmp/cache/ocr_fails.txt")
        if not ocr_fails_log.exists():
            return JSONResponse(content={"failures": []})
        
        with open(ocr_fails_log, 'r', encoding='utf-8') as f:
            failures = f.readlines()[-100:]  # Last 100 failures
        
        return JSONResponse(content={
            "failures": [line.strip() for line in failures],
            "total_shown": len(failures)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read failure logs: {str(e)}")

@app.get("/logs/scanner/")
async def get_scanner_logs():
    """Get recent scanner attempt logs"""
    try:
        import csv
        scanner_log = Path("/tmp/cache/scanner_log.csv")
        if not scanner_log.exists():
            return JSONResponse(content={"logs": []})
        
        logs = []
        with open(scanner_log, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(row)
        
        # Return last 50 entries
        return JSONResponse(content={
            "logs": logs[-50:],
            "total_shown": len(logs[-50:])
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read scanner logs: {str(e)}")

@app.get("/captures/{filename}")
async def get_capture(filename: str):
    """Serve captured images"""
    file_path = CAPTURE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Capture not found")
    return FileResponse(file_path)

@app.get("/debug/images/{filename}")
async def get_debug_image(filename: str):
    """Serve debug images from enhanced OCR processing"""
    file_path = IMAGE_CACHE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Debug image not found")
    return FileResponse(file_path)

@app.get("/health/")
async def health_check():
    """Enhanced health check with cache statistics"""
    try:
        cache_stats = ocr_utils.get_cache_stats()
        return {
            "status": "healthy", 
            "service": "Enhanced MTG Card Scanner Camera API",
            "version": "3.0.0",
            "camera_active": camera_manager.is_active,
            "active_connections": len(active_connections),
            "cache_stats": cache_stats
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "Enhanced MTG Card Scanner Camera API", 
            "version": "3.0.0",
            "camera_active": camera_manager.is_active,
            "active_connections": len(active_connections),
            "cache_error": str(e)
        }

# Mount static files
app.mount("/assets", StaticFiles(directory="/tmp/public/assets"), name="assets")
#app.mount("/assets", StaticFiles(directory=BASE_DIR / "public" / "assets"), name="assets")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")

if __name__ == "__main__":    
    import os
    # Create necessary directories on startup
    os.makedirs("/tmp/public/assets/mindar", exist_ok=True)
    os.makedirs("/tmp/public/assets/models", exist_ok=True)
    os.makedirs("/tmp/cache/images", exist_ok=True)
    os.makedirs("/tmp/cache/captures", exist_ok=True)
    os.makedirs("/tmp/cache/json_cache", exist_ok=True)
    import uvicorn
    print("🚀 Starting Enhanced MTG Card Scanner API v3.0.0")
    print("📊 Features: 5-step OCR pipeline, image hashing, JSON caching, enhanced logging")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)