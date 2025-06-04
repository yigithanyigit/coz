#!/usr/bin/env python3
"""
Chain-of-Zoom FastAPI Server Entry Point
This can be run directly or via pip-installed command: coz-serve
"""
import os
import sys
import io
import base64
import torch
import requests
import tempfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional
from torchvision import transforms

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Chain-of-Zoom components
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import wavelet_color_fix
from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler


# Configuration
MODEL_DIR = Path.home() / ".cache" / "chain-of-zoom"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "lora_path": "ckpt/SR_LoRA/model_20001.pkl",
    "vae_path": "ckpt/SR_VAE/vae_encoder_20001.pt", 
    "ram_path": str(MODEL_DIR / "ram_swin_large_14m.pth"),
    "ram_ft_path": "ckpt/DAPE/DAPE.pth",
    "ram_url": "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth",
}

# FastAPI app
app = FastAPI(title="Chain-of-Zoom API", version="0.1.0")
service = None


class ProcessRequest(BaseModel):
    """API request model"""
    image: str  # Base64 encoded image
    zoom_steps: Optional[int] = 4
    user_prompt: Optional[str] = ""


class ChainOfZoomService:
    """Main service class"""
    
    def __init__(self):
        """Initialize service with automatic model download"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Fix paths to be relative to the package
        base_dir = Path(__file__).parent
        for key in ["lora_path", "vae_path", "ram_ft_path"]:
            CONFIG[key] = str(base_dir / CONFIG[key])
        
        # Transforms
        self.tensor_transforms = transforms.Compose([transforms.ToTensor()])
        self.ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Download and initialize
        self._download_models()
        self._initialize_models()
    
    def _download_models(self):
        """Download missing models"""
        # Download RAM model if needed
        if not os.path.exists(CONFIG["ram_path"]):
            print(f"Downloading RAM model to {CONFIG['ram_path']}...")
            print("This is a one-time download (~5.6GB)")
            
            try:
                response = requests.get(CONFIG["ram_url"], stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(CONFIG["ram_path"], 'wb') as f:
                    with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                print("✓ RAM model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download RAM model: {e}")
        
        # Verify included checkpoint files exist
        for key in ["lora_path", "vae_path", "ram_ft_path"]:
            if not os.path.exists(CONFIG[key]):
                raise FileNotFoundError(
                    f"Missing {key}: {CONFIG[key]}\n"
                    f"This file should be included in the package. "
                    f"Try reinstalling: pip install --force-reinstall chain-of-zoom"
                )
    
    def _initialize_models(self):
        """Initialize all models"""
        print("Initializing Chain-of-Zoom models...")
        
        # Initialize SD3 model
        self.model = SD3Euler()
        
        # Device placement
        if torch.cuda.device_count() > 1:
            self.model.text_enc_1.to('cuda:0')
            self.model.text_enc_2.to('cuda:0')
            self.model.text_enc_3.to('cuda:0')
            self.model.transformer.to('cuda:1', dtype=torch.float32)
            self.model.vae.to('cuda:1', dtype=torch.float32)
        else:
            self.model.text_enc_1.to(self.device)
            self.model.text_enc_2.to(self.device)
            self.model.text_enc_3.to(self.device)
            self.model.transformer.to(self.device, dtype=torch.float32)
            self.model.vae.to(self.device, dtype=torch.float32)
        
        for p in [self.model.text_enc_1, self.model.text_enc_2, self.model.text_enc_3,
                  self.model.transformer, self.model.vae]:
            p.requires_grad_(False)
        
        # Initialize OSEDiff
        class Args:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        args = Args(
            lora_path=CONFIG["lora_path"],
            vae_path=CONFIG["vae_path"],
            lora_rank=4,
            vae_decoder_tiled_size=224,
            vae_encoder_tiled_size=1024,
            latent_tiled_size=96,
            latent_tiled_overlap=32
        )
        self.model_test = OSEDiff_SD3_TEST(args, self.model)
        
        # Initialize DAPE
        self.dape_model = ram(
            pretrained=CONFIG["ram_path"],
            pretrained_condition=CONFIG["ram_ft_path"],
            image_size=384,
            vit='swin_l'
        )
        self.dape_model.eval().to(self.device)
        self.dape_model = self.dape_model.to(dtype=self.weight_dtype)
        
        print("✓ Service ready!")
    
    def process(self, image: Image.Image, zoom_steps: int = 4, user_prompt: str = "") -> Image.Image:
        """Process image through Chain-of-Zoom pipeline"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Initial resize
        current_image = self._resize_and_center_crop(image, 512)
        
        # Recursive zoom
        for step in range(zoom_steps):
            if step > 0:
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
            current_image = wavelet_color_fix(target=sr_output, source=current_image)
        
        return current_image
    
    def _resize_and_center_crop(self, img: Image.Image, size: int) -> Image.Image:
        """Resize and center crop image"""
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))
    
    def _generate_prompt(self, image: Image.Image, user_prompt: str = "") -> tuple:
        """Generate DAPE prompt"""
        lq = self.tensor_transforms(image).unsqueeze(0).to(self.device)
        lq_ram = self.ram_transforms(lq).to(dtype=self.weight_dtype)
        captions = inference(lq_ram, self.dape_model)
        prompt_text = f"{captions[0]}, {user_prompt}," if user_prompt else captions[0]
        return prompt_text, lq
    
    def _apply_super_resolution(self, image_tensor: torch.Tensor, prompt: str) -> Image.Image:
        """Apply super-resolution"""
        with torch.no_grad():
            lq = image_tensor * 2 - 1
            output_image = self.model_test(lq, prompt=prompt)
            output_image = torch.clamp(output_image[0].cpu(), -1.0, 1.0)
            output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)
        return output_pil


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global service
    print("\nChain-of-Zoom FastAPI Server")
    print("="*40)
    try:
        service = ChainOfZoomService()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        raise


@app.post("/process")
async def process_image(request: ProcessRequest):
    """Process an image through Chain-of-Zoom"""
    global service
    
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not 1 <= request.zoom_steps <= 8:
        raise HTTPException(status_code=400, detail="zoom_steps must be between 1 and 8")
    
    try:
        # Decode and process
        image_data = base64.b64decode(request.image)
        input_image = Image.open(io.BytesIO(image_data))
        
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
    """Health check endpoint"""
    global service
    return {
        "status": "healthy" if service else "not_ready",
        "models_loaded": service is not None
    }


def main():
    """Main entry point for console script"""
    import argparse
    parser = argparse.ArgumentParser(description="Chain-of-Zoom FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"Starting server at http://{args.host}:{args.port}")
    print("API docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        "serve:app" if __name__ == "__main__" else "chain_of_zoom.serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()