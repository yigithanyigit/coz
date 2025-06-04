#!/usr/bin/env python3
"""
Flexible Chain-of-Zoom service that preserves aspect ratio and handles arbitrary scales
"""
import os
import sys
import io
import base64
import math
import torch
from pathlib import Path
from PIL import Image
from typing import Optional, Union, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coz_portable_service import ChainOfZoomService as BaseService


class FlexibleChainOfZoomService(BaseService):
    """
    Enhanced Chain-of-Zoom service that handles arbitrary input sizes and scale factors.
    """
    
    def process_image_with_scale(self,
                                image: Union[Image.Image, str, bytes],
                                scale: float,
                                user_prompt: str = "",
                                max_side: int = 2048) -> Image.Image:
        """
        Process an image with a specific scale factor.
        
        Args:
            image: Input image (PIL Image, file path, or bytes)
            scale: Target scale factor (e.g., 2, 3, 4, 8, etc.)
            user_prompt: Optional text prompt
            max_side: Maximum side length to prevent OOM
            
        Returns:
            Upscaled PIL Image with dimensions (w*scale, h*scale)
        """
        # Ensure models are initialized
        if not self.models_initialized:
            self.initialize()
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original dimensions
        orig_w, orig_h = image.size
        target_w = int(orig_w * scale)
        target_h = int(orig_h * scale)
        
        print(f"Input size: {orig_w}x{orig_h}")
        print(f"Target size: {target_w}x{target_h} ({scale}x)")
        
        # Check if target size is too large
        if max(target_w, target_h) > max_side:
            max_scale = max_side / max(orig_w, orig_h)
            print(f"Warning: Target size too large. Limiting scale to {max_scale:.2f}x")
            scale = max_scale
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)
        
        # Calculate zoom steps needed
        # Each zoom step does 4x, so we need ceil(log4(scale)) steps
        zoom_steps = max(1, math.ceil(math.log(scale, 4)))
        zoom_steps = min(zoom_steps, 8)  # Max 8 steps
        
        # Actual scale achieved with zoom_steps
        achieved_scale = 4 ** zoom_steps
        
        print(f"Using {zoom_steps} zoom steps (achieves {achieved_scale}x internally)")
        
        # Process at original aspect ratio
        current_image = self._process_at_original_ratio(image, zoom_steps, user_prompt)
        
        # Resize to exact target dimensions if needed
        if current_image.size != (target_w, target_h):
            current_image = current_image.resize((target_w, target_h), Image.LANCZOS)
            print(f"Final resize to exact target: {target_w}x{target_h}")
        
        return current_image
    
    def _process_at_original_ratio(self, image: Image.Image, zoom_steps: int, user_prompt: str) -> Image.Image:
        """Process image while maintaining aspect ratio."""
        orig_w, orig_h = image.size
        
        # Determine processing size that maintains aspect ratio
        # Use 512 as base, but adjust to maintain ratio
        if orig_w > orig_h:
            process_w = 512
            process_h = int(512 * orig_h / orig_w)
        else:
            process_h = 512
            process_w = int(512 * orig_w / orig_h)
        
        # Round to multiple of 8 for better compatibility
        process_w = (process_w // 8) * 8
        process_h = (process_h // 8) * 8
        
        print(f"Processing at: {process_w}x{process_h}")
        
        # Initial resize
        current_image = image.resize((process_w, process_h), Image.LANCZOS)
        
        # Recursive zoom processing
        for step in range(zoom_steps):
            print(f"Processing zoom step {step + 1}/{zoom_steps}...")
            
            if step > 0:
                # Crop center and upscale for next iteration
                w, h = current_image.size
                new_w, new_h = w // 4, h // 4
                cropped = current_image.crop(
                    ((w - new_w) // 2, (h - new_h) // 2,
                     (w + new_w) // 2, (h + new_h) // 2)
                )
                current_image = cropped.resize((w, h), Image.BICUBIC)
            
            # Generate prompt and apply SR
            prompt_text, lq = self._generate_prompt(current_image, user_prompt)
            sr_output = self._apply_super_resolution(lq, prompt_text)
            
            # Import wavelet fix
            from utils.wavelet_color_fix import wavelet_color_fix
            current_image = wavelet_color_fix(target=sr_output, source=current_image)
        
        # Scale back to original aspect ratio at higher resolution
        final_w = orig_w * (4 ** zoom_steps)
        final_h = orig_h * (4 ** zoom_steps)
        current_image = current_image.resize((final_w, final_h), Image.LANCZOS)
        
        return current_image


def create_flexible_api():
    """Create FastAPI app with flexible scaling."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="Flexible Chain-of-Zoom API", version="0.1.0")
    service = FlexibleChainOfZoomService()
    
    class ProcessRequest(BaseModel):
        image: str  # Base64 encoded image
        scale: float = 4.0  # Target scale factor
        user_prompt: Optional[str] = ""
        max_side: Optional[int] = 2048
    
    @app.post("/upscale")
    async def upscale_image(request: ProcessRequest):
        try:
            # Decode image
            image_data = base64.b64decode(request.image)
            input_image = Image.open(io.BytesIO(image_data))
            orig_w, orig_h = input_image.size
            
            # Process with scale
            output_image = service.process_image_with_scale(
                image=image_data,
                scale=request.scale,
                user_prompt=request.user_prompt,
                max_side=request.max_side
            )
            
            # Encode result
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            buffer.seek(0)
            result_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "success": True,
                "image": result_base64,
                "original_size": [orig_w, orig_h],
                "output_size": list(output_image.size),
                "scale": request.scale
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {
            "message": "Flexible Chain-of-Zoom API",
            "endpoints": {
                "/upscale": "Upscale image by arbitrary scale factor",
                "/docs": "API documentation"
            }
        }
    
    return app


# Example client code
def example_client():
    """Example of calling the flexible API"""
    import requests
    
    # Read image
    with open("samples/0064.png", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # Call API with custom scale
    response = requests.post(
        "http://localhost:8000/upscale",
        json={
            "image": image_base64,
            "scale": 3.0,  # 3x upscale
            "user_prompt": "highly detailed"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Original size: {result['original_size']}")
        print(f"Output size: {result['output_size']}")
        print(f"Scale: {result['scale']}")
        
        # Save result
        image_data = base64.b64decode(result['image'])
        Image.open(io.BytesIO(image_data)).save("upscaled_3x.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--example", action="store_true", help="Run example client")
    args = parser.parse_args()
    
    if args.serve:
        import uvicorn
        app = create_flexible_api()
        print("Starting Flexible Chain-of-Zoom API...")
        print("Endpoint: POST http://localhost:8000/upscale")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.example:
        example_client()
    else:
        print("Usage:")
        print("  python coz_flexible_service.py --serve    # Start server")
        print("  python coz_flexible_service.py --example  # Run client example")