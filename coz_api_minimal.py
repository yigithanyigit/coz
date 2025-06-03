import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import Optional

# Import our unified service that encapsulates all CoZ logic
from coz_service_unified import ChainOfZoomService


app = FastAPI(title="Chain-of-Zoom API", version="1.0.0")

# Global service instance
service = None


class ProcessRequest(BaseModel):
    """Request model matching inference_coz.py CLI arguments"""
    image: str  # Base64 encoded image
    zoom_steps: Optional[int] = 4  # --rec_num in CLI (default: 4)
    user_prompt: Optional[str] = ""  # --prompt in CLI


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup.
    
    This replaces the model initialization section in inference_coz.py:172-231
    but loads models once at startup instead of on each run.
    """
    global service
    print("Starting Chain-of-Zoom service...")
    service = ChainOfZoomService()
    print("Service ready!")


@app.post("/process")
async def process_image(request: ProcessRequest):
    """
    Process an image through Chain-of-Zoom.
    
    Args:
        image: Base64 encoded PNG/JPEG image
        zoom_steps: Number of zoom iterations (1-8, default: 4)
        user_prompt: Optional text to append to auto-generated prompts
    
    Returns:
        Base64 encoded result image
    """
    global service
    
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate zoom steps
    if not 1 <= request.zoom_steps <= 8:
        raise HTTPException(status_code=400, detail="zoom_steps must be between 1 and 8")
    
    try:
        # Decode input image
        image_data = base64.b64decode(request.image)
        input_image = Image.open(io.BytesIO(image_data))
        
        # Process image - this replaces the main inference loop
        # from inference_coz.py:241-364
        output_image = service.process(
            image=input_image,
            zoom_steps=request.zoom_steps,
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
            "zoom_steps": request.zoom_steps,
            "final_resolution": output_image.size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
async def health_check():
    """Check if service is healthy and models are loaded."""
    global service
    return {
        "status": "healthy" if service else "not_ready",
        "models_loaded": service is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)