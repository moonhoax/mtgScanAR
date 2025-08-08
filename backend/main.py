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
from backend import ocr_utils
import asyncio
from datetime import datetime

app = FastAPI(
    title="MTG Card Scanner Camera API",
    description="Real-time Magic: The Gathering card scanning with camera support and AR integration",
    version="2.0.0"
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
IMAGE_CACHE_DIR = BASE_DIR / "cache"
CAPTURE_DIR = BASE_DIR / "cache" / "captures"
IMAGE_CACHE_DIR.mkdir(exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)

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
    Scan a Magic: The Gathering card image and return AR-ready data
    Enhanced with automatic card detection and perspective correction
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        await file.close()
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded image to cache
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{file.filename}"
    file_path = IMAGE_CACHE_DIR / filename
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        await file.close()
    except Exception as e:
        await file.close()
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    try:
        # Run the enhanced processing pipeline
        result = ocr_utils.process_camera_capture(str(file_path))
        
        # Add processing metadata
        result['processing_timestamp'] = timestamp
        result['source'] = 'file_upload'
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up on error
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/camera/capture/")
async def capture_from_camera():
    """
    Capture and process a card image directly from device camera
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
        # Process the captured image
        result = ocr_utils.process_camera_capture(str(file_path))
        
        # Add processing metadata
        result['processing_timestamp'] = timestamp
        result['source'] = 'camera_capture'
        result['captured_image'] = f"/captures/{filename}"
        
        return JSONResponse(content=result)
        
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Camera processing error: {str(e)}")

@app.websocket("/ws/camera-stream/")
async def camera_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera streaming and card detection
    """
    if len(active_connections) >= MAX_WEBSOCKET_CLIENTS:
        await websocket.close(code=1008, reason="Max connections reached")
        return

    await websocket.accept()
    active_connections.append(websocket)
    
    print("📱 Camera stream WebSocket connected")
    
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
                        # Process current frame for card detection
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"stream_capture_{timestamp}.jpg"
                        file_path = CAPTURE_DIR / filename
                        
                        cv2.imwrite(str(file_path), frame)
                        
                        try:
                            result = ocr_utils.process_camera_capture(str(file_path))
                            result['processing_timestamp'] = timestamp
                            result['source'] = 'camera_stream'
                            result['captured_image'] = f"/captures/{filename}"
                            
                            await websocket.send_json({
                                "type": "scan_result",
                                **result
                            })
                            
                        except Exception as e:
                            await websocket.send_json({
                                "type": "scan_error",
                                "error": str(e)
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
        print("📱 Camera stream WebSocket disconnected")
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
    """Scan multiple card images at once"""
    if len(files) > 10:  # Reasonable limit for camera captures
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
            
            result = ocr_utils.process_camera_capture(str(file_path))
            result['filename'] = file.filename
            result['processing_timestamp'] = timestamp
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': f'Processing failed: {str(e)}'
            })
        finally:
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

@app.get("/captures/{filename}")
async def get_capture(filename: str):
    """Serve captured images"""
    file_path = CAPTURE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Capture not found")
    return FileResponse(file_path)

@app.get("/debug/images/{filename}")
async def get_debug_image(filename: str):
    """Serve debug images from OCR processing"""
    file_path = IMAGE_CACHE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Debug image not found")
    return FileResponse(file_path)

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "MTG Card Scanner Camera API",
        "camera_active": camera_manager.is_active,
        "active_connections": len(active_connections)
    }

# Mount static files
app.mount("/assets", StaticFiles(directory=BASE_DIR / "public" / "assets"), name="assets")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
