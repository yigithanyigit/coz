#!/usr/bin/env python3
"""
Portable Chain-of-Zoom FastAPI Service

This is a self-contained service that can be run standalone:
    python coz_portable_service.py

Or imported and used as a class:
    from coz_portable_service import ChainOfZoomService
    service = ChainOfZoomService()
    result = service.process_image(input_pil_image)
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
from typing import Optional, Union
from torchvision import transforms

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Chain-of-Zoom components
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import wavelet_color_fix
from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler


class ChainOfZoomService:
    """
    Self-contained Chain-of-Zoom service class.
    
    This class handles:
    - Automatic model downloads
    - Model initialization
    - Image processing through the Chain-of-Zoom pipeline
    
    Usage:
        service = ChainOfZoomService()
        output_image = service.process_image(input_pil_image, zoom_steps=4)
    """
    
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 download_on_init: bool = True):
        """
        Initialize the Chain-of-Zoom service.
        
        Args:
            model_dir: Directory for downloaded models (default: ~/.cache/chain-of-zoom)
            device: Device to use ('cuda' or 'cpu', default: auto-detect)
            download_on_init: Whether to download missing models during init
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        # Set model directory
        self.model_dir = Path(model_dir) if model_dir else (Path.home() / ".cache" / "chain-of-zoom")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.base_dir = Path(__file__).parent
        self.config = {
            "lora_path": self.base_dir / "ckpt/SR_LoRA/model_20001.pkl",
            "vae_path": self.base_dir / "ckpt/SR_VAE/vae_encoder_20001.pt",
            "ram_path": self.model_dir / "ram_swin_large_14m.pth",
            "ram_ft_path": self.base_dir / "ckpt/DAPE/DAPE.pth",
            "ram_url": "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth",
        }
        
        # Initialize transforms
        self.tensor_transforms = transforms.Compose([transforms.ToTensor()])
        self.ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models if requested
        self.models_initialized = False
        if download_on_init:
            self.initialize()
    
    def download_ram_model(self) -> bool:
        """Download RAM model if not present."""
        ram_path = self.config["ram_path"]
        
        if ram_path.exists():
            return True
        
        print(f"Downloading RAM model to {ram_path}...")
        print("This is a one-time download (~5.6GB)")
        
        try:
            response = requests.get(self.config["ram_url"], stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            # Download to temp file first
            temp_path = ram_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc="RAM model") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Move to final location
            temp_path.rename(ram_path)
            print("✓ RAM model downloaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download RAM model: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def verify_checkpoints(self) -> bool:
        """Verify all required checkpoint files exist."""
        required_files = [
            (self.config["lora_path"], "SR LoRA weights"),
            (self.config["vae_path"], "VAE encoder weights"),
            (self.config["ram_ft_path"], "DAPE weights"),
        ]
        
        all_present = True
        for path, desc in required_files:
            if not path.exists():
                print(f"✗ Missing {desc}: {path}")
                all_present = False
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ Found {desc}: {path.name} ({size_mb:.1f} MB)")
        
        return all_present
    
    def initialize(self):
        """Initialize all models. Called automatically on first use if not called manually."""
        if self.models_initialized:
            return
        
        print("\nInitializing Chain-of-Zoom models...")
        
        # Download RAM if needed
        if not self.download_ram_model():
            raise RuntimeError("Failed to download RAM model")
        
        # Verify checkpoints
        if not self.verify_checkpoints():
            raise RuntimeError("Missing required checkpoint files")
        
        # Initialize SD3 model
        print("Loading SD3 model...")
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
        
        # Disable gradients
        for p in [self.model.text_enc_1, self.model.text_enc_2, self.model.text_enc_3,
                  self.model.transformer, self.model.vae]:
            p.requires_grad_(False)
        
        # Initialize OSEDiff
        print("Loading LoRA weights...")
        class Args:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        args = Args(
            lora_path=str(self.config["lora_path"]),
            vae_path=str(self.config["vae_path"]),
            lora_rank=4,
            vae_decoder_tiled_size=224,
            vae_encoder_tiled_size=1024,
            latent_tiled_size=96,
            latent_tiled_overlap=32
        )
        self.model_test = OSEDiff_SD3_TEST(args, self.model)
        
        # Initialize DAPE
        print("Loading DAPE model...")
        self.dape_model = ram(
            pretrained=str(self.config["ram_path"]),
            pretrained_condition=str(self.config["ram_ft_path"]),
            image_size=384,
            vit='swin_l'
        )
        self.dape_model.eval().to(self.device)
        self.dape_model = self.dape_model.to(dtype=self.weight_dtype)
        
        self.models_initialized = True
        print("✓ All models loaded successfully!")
    
    def process_image(self, 
                      image: Union[Image.Image, str, bytes],
                      zoom_steps: int = 4,
                      user_prompt: str = "") -> Image.Image:
        """
        Process an image through the Chain-of-Zoom pipeline.
        
        Args:
            image: PIL Image, file path, or bytes
            zoom_steps: Number of zoom iterations (1-8)
            user_prompt: Optional text prompt to append
            
        Returns:
            Processed PIL Image
        """
        # Ensure models are initialized
        if not self.models_initialized:
            self.initialize()
        
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        # Validate parameters
        if not 1 <= zoom_steps <= 8:
            raise ValueError("zoom_steps must be between 1 and 8")
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Initial resize
        current_image = self._resize_and_center_crop(image, 512)
        
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
            current_image = wavelet_color_fix(target=sr_output, source=current_image)
        
        return current_image
    
    def _resize_and_center_crop(self, img: Image.Image, size: int) -> Image.Image:
        """Resize and center crop image."""
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))
    
    def _generate_prompt(self, image: Image.Image, user_prompt: str = "") -> tuple:
        """Generate DAPE prompt for the image."""
        lq = self.tensor_transforms(image).unsqueeze(0).to(self.device)
        lq_ram = self.ram_transforms(lq).to(dtype=self.weight_dtype)
        captions = inference(lq_ram, self.dape_model)
        prompt_text = f"{captions[0]}, {user_prompt}," if user_prompt else captions[0]
        return prompt_text, lq
    
    def _apply_super_resolution(self, image_tensor: torch.Tensor, prompt: str) -> Image.Image:
        """Apply super-resolution to the image tensor."""
        with torch.no_grad():
            lq = image_tensor * 2 - 1
            output_image = self.model_test(lq, prompt=prompt)
            output_image = torch.clamp(output_image[0].cpu(), -1.0, 1.0)
            output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)
        return output_pil


# Optional: FastAPI integration
def create_fastapi_app(service: Optional[ChainOfZoomService] = None):
    """Create a FastAPI app with the Chain-of-Zoom service."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="Chain-of-Zoom API", version="0.1.0")
    
    # Use provided service or create new one
    if service is None:
        service = ChainOfZoomService()
    
    class ProcessRequest(BaseModel):
        image: str  # Base64 encoded image
        zoom_steps: Optional[int] = 4
        user_prompt: Optional[str] = ""
    
    @app.post("/process")
    async def process_image(request: ProcessRequest):
        try:
            # Decode base64 image
            image_data = base64.b64decode(request.image)
            
            # Process image
            output_image = service.process_image(
                image=image_data,
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
        return {
            "status": "healthy",
            "models_loaded": service.models_initialized
        }
    
    return app


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain-of-Zoom Service")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for server")
    parser.add_argument("--port", type=int, default=8000, help="Port for server")
    parser.add_argument("--input", type=str, help="Process a single image")
    parser.add_argument("--output", type=str, help="Output path for processed image")
    parser.add_argument("--zoom-steps", type=int, default=4, help="Number of zoom steps")
    parser.add_argument("--prompt", type=str, default="", help="User prompt")
    
    args = parser.parse_args()
    
    # Create service
    print("Initializing Chain-of-Zoom service...")
    service = ChainOfZoomService()
    
    if args.serve:
        # Start FastAPI server
        import uvicorn
        app = create_fastapi_app(service)
        print(f"\nStarting server at http://{args.host}:{args.port}")
        print(f"API docs: http://localhost:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port)
        
    elif args.input:
        # Process single image
        if not args.output:
            base_name = Path(args.input).stem
            args.output = f"{base_name}_coz_z{args.zoom_steps}.png"
        
        print(f"\nProcessing {args.input}...")
        output_image = service.process_image(
            args.input,
            zoom_steps=args.zoom_steps,
            user_prompt=args.prompt
        )
        output_image.save(args.output)
        print(f"✓ Saved to {args.output}")
        
    else:
        # Interactive mode
        print("\nChain-of-Zoom service ready!")
        print("\nUsage examples:")
        print("  # Start API server:")
        print(f"  python {__file__} --serve")
        print("\n  # Process an image:")
        print(f"  python {__file__} --input image.png --output result.png --zoom-steps 4")
        print("\n  # Use as a library:")
        print("  from coz_portable_service import ChainOfZoomService")
        print("  service = ChainOfZoomService()")
        print("  output = service.process_image('image.png', zoom_steps=4)")