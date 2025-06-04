#!/usr/bin/env python3
"""
Chain-of-Zoom Service - Extreme zoom into image regions

This service implements the actual Chain-of-Zoom method:
zooming into a specific region of an image to reveal details,
NOT upscaling the entire image.
"""
import os
import sys
import io
import base64
import torch
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union, Tuple, List
from torchvision import transforms

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Chain-of-Zoom components
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import wavelet_color_fix
from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler


class ChainOfZoomService:
    """
    Chain-of-Zoom service for extreme zoom into image regions.
    
    Each zoom step:
    1. Crops the center region (1/4 of area)
    2. Upscales it back to original size (4x zoom)
    3. Applies SR with context-aware prompts
    """
    
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 process_size: int = 512):
        """
        Initialize Chain-of-Zoom service.
        
        Args:
            model_dir: Directory for downloaded models
            device: Device to use ('cuda' or 'cpu')
            process_size: Processing resolution (default: 512)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.process_size = process_size
        
        # Model directory
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
        
        # Transforms
        self.tensor_transforms = transforms.Compose([transforms.ToTensor()])
        self.ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.models_initialized = False
    
    def initialize(self):
        """Initialize models (called automatically on first use)."""
        if self.models_initialized:
            return
        
        print("Initializing Chain-of-Zoom models...")
        
        # Download RAM if needed
        if not self.config["ram_path"].exists():
            self._download_ram_model()
        
        # Initialize models
        self._initialize_sd3()
        self._initialize_osediff()
        self._initialize_dape()
        
        self.models_initialized = True
        print("✓ Models initialized successfully!")
    
    def zoom(self,
             image: Union[Image.Image, str, bytes],
             zoom_steps: int = 4,
             center: Optional[Tuple[float, float]] = None,
             user_prompt: str = "",
             return_intermediate: bool = False) -> Union[Image.Image, List[Image.Image]]:
        """
        Perform extreme zoom into an image region.
        
        Args:
            image: Input image (PIL Image, file path, or bytes)
            zoom_steps: Number of zoom iterations (1-8)
            center: Zoom center as (x, y) in [0, 1] coordinates. None = center
            user_prompt: Optional text to append to prompts
            return_intermediate: If True, return all intermediate zoom levels
            
        Returns:
            Final zoomed image or list of all zoom levels
        """
        # Initialize models if needed
        if not self.models_initialized:
            self.initialize()
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to process size
        current_image = self._resize_and_center_crop(image, self.process_size)
        
        # Default center
        if center is None:
            center = (0.5, 0.5)
        
        # Track zoom region for visualization
        zoom_regions = [(0, 0, 1, 1)]  # Full image
        results = [current_image.copy()] if return_intermediate else []
        
        # Zoom iterations
        for step in range(zoom_steps):
            print(f"Zoom step {step + 1}/{zoom_steps} (total zoom: {4**(step+1)}x)...")
            
            # Calculate zoom region
            if step > 0:
                # Get the region to zoom into (1/4 of current area = 1/2 width, 1/2 height)
                w, h = current_image.size
                crop_w, crop_h = w // 2, h // 2
                
                # Calculate crop bounds centered at specified point
                cx = int(center[0] * w)
                cy = int(center[1] * h)
                
                # Ensure crop stays within bounds
                left = max(0, min(w - crop_w, cx - crop_w // 2))
                top = max(0, min(h - crop_h, cy - crop_h // 2))
                right = left + crop_w
                bottom = top + crop_h
                
                # Crop and resize back to process size
                cropped = current_image.crop((left, top, right, bottom))
                current_image = cropped.resize((w, h), Image.BICUBIC)
                
                # Update zoom region tracking
                prev_region = zoom_regions[-1]
                region_w = prev_region[2] - prev_region[0]
                region_h = prev_region[3] - prev_region[1]
                new_region = (
                    prev_region[0] + (left / w) * region_w,
                    prev_region[1] + (top / h) * region_h,
                    prev_region[0] + (right / w) * region_w,
                    prev_region[1] + (bottom / h) * region_h
                )
                zoom_regions.append(new_region)
            
            # Generate prompt and apply SR
            prompt_text = self._generate_prompt(current_image, user_prompt, step + 1)
            current_image = self._apply_sr_step(current_image, prompt_text)
            
            if return_intermediate:
                results.append(current_image.copy())
        
        return results if return_intermediate else current_image
    
    def _generate_prompt(self, image: Image.Image, user_prompt: str, zoom_level: int) -> str:
        """Generate context-aware prompt for current zoom level."""
        # Get DAPE caption
        lq = self.tensor_transforms(image).unsqueeze(0).to(self.device)
        lq_ram = self.ram_transforms(lq).to(dtype=self.weight_dtype)
        captions = inference(lq_ram, self.dape_model)
        
        # Add zoom context
        zoom_factor = 4 ** zoom_level
        zoom_context = f"zoomed {zoom_factor}x view showing fine details"
        
        # Combine prompts
        base_prompt = captions[0]
        if user_prompt:
            prompt_text = f"{base_prompt}, {user_prompt}, {zoom_context}"
        else:
            prompt_text = f"{base_prompt}, {zoom_context}"
        
        return prompt_text
    
    def _apply_sr_step(self, image: Image.Image, prompt: str) -> Image.Image:
        """Apply one super-resolution step."""
        # Convert to tensor
        lq = self.tensor_transforms(image).unsqueeze(0).to(self.device)
        
        # Apply SR
        with torch.no_grad():
            lq = lq * 2 - 1
            output = self.model_test(lq, prompt=prompt)
            output = torch.clamp(output[0].cpu(), -1.0, 1.0)
            output_pil = transforms.ToPILImage()(output * 0.5 + 0.5)
        
        # Apply color correction
        output_pil = wavelet_color_fix(target=output_pil, source=image)
        
        return output_pil
    
    
    def _resize_and_center_crop(self, img: Image.Image, size: int) -> Image.Image:
        """Resize and center crop image."""
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))
    
    def _download_ram_model(self):
        """Download RAM model."""
        print(f"Downloading RAM model (~5.6GB)...")
        try:
            response = requests.get(self.config["ram_url"], stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            temp_path = self.config["ram_path"].with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            temp_path.rename(self.config["ram_path"])
            print("✓ RAM model downloaded")
        except Exception as e:
            raise RuntimeError(f"Failed to download RAM model: {e}")
    
    def _initialize_sd3(self):
        """Initialize SD3 model."""
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
        
        for p in [self.model.text_enc_1, self.model.text_enc_2, self.model.text_enc_3,
                  self.model.transformer, self.model.vae]:
            p.requires_grad_(False)
    
    def _initialize_osediff(self):
        """Initialize OSEDiff with LoRA."""
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
    
    def _initialize_dape(self):
        """Initialize DAPE model."""
        print("Loading DAPE model...")
        self.dape_model = ram(
            pretrained=str(self.config["ram_path"]),
            pretrained_condition=str(self.config["ram_ft_path"]),
            image_size=384,
            vit='swin_l'
        )
        self.dape_model.eval().to(self.device)
        self.dape_model = self.dape_model.to(dtype=self.weight_dtype)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain-of-Zoom: Extreme zoom into image regions")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--zoom-steps", type=int, default=4, help="Number of zoom steps (1-8)")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--center", nargs=2, type=float, metavar=('X', 'Y'), 
                       help="Zoom center (0-1, default: 0.5 0.5)")
    parser.add_argument("--save-intermediate", action="store_true", 
                       help="Save all intermediate zoom levels")
    
    args = parser.parse_args()
    
    # Initialize service
    print("Initializing Chain-of-Zoom service...")
    service = ChainOfZoomService()
    
    # Set zoom center
    center = tuple(args.center) if args.center else None
    
    # Perform zoom
    result = service.zoom(
        args.input,
        zoom_steps=args.zoom_steps,
        center=center,
        return_intermediate=args.save_intermediate
    )
    
    # Save results
    if args.save_intermediate:
        base_name = Path(args.input).stem
        for i, img in enumerate(result):
            zoom_factor = 4 ** i
            output_path = f"{base_name}_zoom_{zoom_factor}x.png"
            img.save(output_path)
            print(f"Saved: {output_path}")
    else:
        output_path = args.output or f"{Path(args.input).stem}_zoom_{4**args.zoom_steps}x.png"
        result.save(output_path)
        print(f"Saved: {output_path}")