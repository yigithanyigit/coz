#!/usr/bin/env python3
"""
Chain-of-Zoom FastAPI service for extreme zoom functionality
"""
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import Optional, List

from coz_zoom_service import ChainOfZoomService


app = FastAPI(title="Chain-of-Zoom API", version="1.0.0")
service = None


class ZoomRequest(BaseModel):
    """Request model for zoom endpoint"""
    image: str  # Base64 encoded image
    zoom_steps: Optional[int] = 4  # Number of zoom iterations (1-8)
    center_x: Optional[float] = 0.5  # X coordinate (0-1)
    center_y: Optional[float] = 0.5  # Y coordinate (0-1)
    user_prompt: Optional[str] = ""  # Additional prompt text


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global service
    print("Starting Chain-of-Zoom service...")
    try:
        service = ChainOfZoomService()
        service.initialize()
        print("✓ Service ready!")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        raise


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Chain-of-Zoom API",
        "description": "Extreme zoom into image regions",
        "endpoints": {
            "/zoom": "Zoom into a specific region of an image",
            "/docs": "API documentation"
        },
        "usage": {
            "zoom_steps": "Each step zooms 4x deeper (1=4x, 2=16x, 3=64x, 4=256x)",
            "center": "Specify (center_x, center_y) in 0-1 range to zoom into specific region"
        }
    }


@app.post("/zoom")
async def zoom_image(request: ZoomRequest):
    """
    Perform extreme zoom into an image region.
    
    Each zoom step:
    - Crops the center 1/4 area (1/2 width, 1/2 height)
    - Upscales it back to original size (4x zoom)
    - Applies AI super-resolution with context-aware prompts
    """
    global service
    
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate parameters
    if not 1 <= request.zoom_steps <= 8:
        raise HTTPException(status_code=400, detail="zoom_steps must be between 1 and 8")
    
    if not (0 <= request.center_x <= 1 and 0 <= request.center_y <= 1):
        raise HTTPException(status_code=400, detail="center coordinates must be between 0 and 1")
    
    try:
        # Decode image
        image_data = base64.b64decode(request.image)
        input_image = Image.open(io.BytesIO(image_data))
        
        # Perform zoom
        output_image = service.zoom(
            image=input_image,
            zoom_steps=request.zoom_steps,
            center=(request.center_x, request.center_y),
            user_prompt=request.user_prompt
        )
        
        # Encode result
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "image": result_base64,
            "zoom_factor": 4 ** request.zoom_steps,
            "zoom_center": [request.center_x, request.center_y]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global service
    return {
        "status": "healthy" if service and service.models_initialized else "not_ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)