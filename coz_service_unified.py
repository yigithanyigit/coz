import os
import sys
sys.path.append(os.getcwd())
import torch
from torchvision import transforms
from PIL import Image
from typing import Optional
import tempfile
import requests
from tqdm import tqdm

# Original imports from inference_coz.py:12-14
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

# Original imports from inference_coz.py:176,189
from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler


class ChainOfZoomService:
    """
    Unified Chain-of-Zoom service with recommended preset.
    Default configuration: DAPE prompts with recursive mode.
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize the service with recommended configuration."""
        self.device = device
        
        # Fixed recommended configuration
        # From scripts/inference/inference_coz_dapeprompt.sh:12-16
        self.lora_path = "ckpt/SR_LoRA/model_20001.pkl"
        self.vae_path = "ckpt/SR_VAE/vae_encoder_20001.pt"
        self.ram_path = "ckpt/RAM/ram_swin_large_14m.pth"
        self.ram_ft_path = "ckpt/DAPE/DAPE.pth"
        self.pretrained_model_name = "stabilityai/stable-diffusion-3-medium-diffusers"
        
        # Fixed parameters for best quality
        # Default values from inference_coz.py:144-157
        self.process_size = 512
        self.upscale = 4
        self.lora_rank = 4
        self.weight_dtype = torch.float16  # inference_coz.py:169-170
        
        # Initialize transforms
        # From inference_coz.py:16-22
        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models
        self._initialize_models()
        
    def _download_file_with_progress(self, url: str, dest_path: str) -> bool:
        """Download a file with progress bar if it doesn't exist."""
        if os.path.exists(dest_path):
            return True
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"Downloading {os.path.basename(dest_path)}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Downloaded {os.path.basename(dest_path)}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {os.path.basename(dest_path)}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    
    def _ensure_models_available(self):
        """Ensure all required models are available, downloading if necessary."""
        # Check and download RAM model if needed
        ram_path = self.ram_path
        if not os.path.exists(ram_path):
            print(f"\nRAM model not found at {ram_path}")
            print("Downloading from HuggingFace (this may take a while, ~5.6GB)...")
            
            ram_url = "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"
            if not self._download_file_with_progress(ram_url, ram_path):
                raise RuntimeError(
                    f"Failed to download RAM model. Please manually download from:\n"
                    f"{ram_url}\n"
                    f"and save to: {ram_path}"
                )
        
        # Verify other required files exist (already in repo)
        required_files = [
            (self.lora_path, "SR LoRA weights"),
            (self.vae_path, "VAE encoder weights"),
            (self.ram_ft_path, "DAPE weights"),
        ]
        
        for path, desc in required_files:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{desc} not found at {path}. "
                    f"This file should be included in the repository. "
                    f"Please ensure you have cloned the complete repository."
                )
    
    def _initialize_models(self):
        """Initialize all required models."""
        # Ensure models are available first
        self._ensure_models_available()
        
        print("Initializing Chain-of-Zoom models...")
        
        # Initialize SR model
        # From inference_coz.py:177 - SD3Euler initialization
        print("Loading SR model...")
        self.model = SD3Euler()
        
        # Device placement for multi-GPU or single GPU
        # From inference_coz.py:178-182 (multi-GPU setup)
        if torch.cuda.device_count() > 1:
            self.model.text_enc_1.to('cuda:0')
            self.model.text_enc_2.to('cuda:0')
            self.model.text_enc_3.to('cuda:0')
            self.model.transformer.to('cuda:1', dtype=torch.float32)
            self.model.vae.to('cuda:1', dtype=torch.float32)
        else:
            # Single GPU fallback (inference_coz.py:191-192 for efficient mode)
            self.model.text_enc_1.to('cuda')
            self.model.text_enc_2.to('cuda')
            self.model.text_enc_3.to('cuda')
            self.model.transformer.to('cuda', dtype=torch.float32)
            self.model.vae.to('cuda', dtype=torch.float32)
        
        # Disable gradients
        # From inference_coz.py:183-184
        for p in [self.model.text_enc_1, self.model.text_enc_2, self.model.text_enc_3, 
                  self.model.transformer, self.model.vae]:
            p.requires_grad_(False)
        
        # Create args for model initialization
        # Args pattern used throughout inference_coz.py for passing parameters
        class Args:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        args = Args(
            lora_path=self.lora_path,
            vae_path=self.vae_path,
            lora_rank=self.lora_rank,
            # Tiling parameters from inference_coz.py:158-161
            vae_decoder_tiled_size=224,
            vae_encoder_tiled_size=1024,
            latent_tiled_size=96,
            latent_tiled_overlap=32
        )
        
        # Initialize OSEDiff model
        # From inference_coz.py:185 or 195 depending on efficient_memory
        self.model_test = OSEDiff_SD3_TEST(args, self.model)
        
        # Load DAPE model
        # From inference_coz.py:206-212
        print("Loading DAPE model...")
        self.dape_model = ram(
            pretrained=self.ram_path,
            pretrained_condition=self.ram_ft_path,
            image_size=384,
            vit='swin_l'
        )
        self.dape_model.eval().to(self.device)
        self.dape_model = self.dape_model.to(dtype=self.weight_dtype)
        
        print("Service initialization complete!")
    
    def _resize_and_center_crop(self, img: Image.Image, size: int) -> Image.Image:
        """Resize and center crop an image."""
        # From inference_coz.py:24-31 - resize_and_center_crop()
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))
    
    def _generate_prompt(self, image: Image.Image, user_prompt: str = "") -> tuple:
        """Generate prompt using DAPE model."""
        # From inference_coz.py:33-42 - get_validation_prompt() with prompt_type=="dape"
        lq = self.tensor_transforms(image).unsqueeze(0).to(self.device)
        lq_ram = self.ram_transforms(lq).to(dtype=self.weight_dtype)
        captions = inference(lq_ram, self.dape_model)  # inference_coz.py:41
        prompt_text = f"{captions[0]}, {user_prompt}," if user_prompt else captions[0]  # inference_coz.py:42
        return prompt_text, lq
    
    def _apply_super_resolution(self, image_tensor: torch.Tensor, prompt: str) -> Image.Image:
        """Apply super-resolution to the image tensor."""
        # From inference_coz.py:332-347 - super-resolution section
        with torch.no_grad():
            # Normalize input - inference_coz.py:333
            lq = image_tensor * 2 - 1
            
            # Run SR model - inference_coz.py:345
            output_image = self.model_test(lq, prompt=prompt)
            output_image = torch.clamp(output_image[0].cpu(), -1.0, 1.0)  # inference_coz.py:346
            
            # Convert to PIL - inference_coz.py:347
            output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)
            
        return output_pil
    
    def _recursive_zoom_step(self, current_image: Image.Image, user_prompt: str = "") -> Image.Image:
        """Perform one recursive zoom step."""
        # Generate prompt for current image
        # Corresponds to inference_coz.py:325
        prompt_text, lq = self._generate_prompt(current_image, user_prompt)
        
        # Apply super-resolution
        sr_output = self._apply_super_resolution(lq, prompt_text)
        
        # Apply wavelet color fix for better quality
        # From inference_coz.py:351 - wavelet color fix application
        sr_output = wavelet_color_fix(target=sr_output, source=current_image)
        
        return sr_output
    
    def process(self, 
                image: Image.Image, 
                zoom_steps: int = 4,
                user_prompt: str = "",
                return_intermediate: bool = False) -> Image.Image:
        """
        Process an image through the Chain-of-Zoom pipeline.
        
        Args:
            image: Input PIL image
            zoom_steps: Number of zoom recursions (default: 4)
            user_prompt: Optional user prompt to append
            return_intermediate: If True, return list of all intermediate images
            
        Returns:
            Processed PIL image at final zoom level (or list if return_intermediate=True)
        """
        # Convert to RGB if necessary
        # From inference_coz.py:252
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Initial resize and crop
        # From inference_coz.py:253
        current_image = self._resize_and_center_crop(image, self.process_size)
        
        # Store intermediate results if requested
        if return_intermediate:
            results = [current_image.copy()]
        
        # Recursive zoom processing
        # Main recursion loop from inference_coz.py:258
        for step in range(zoom_steps):
            # For recursive mode: crop center and upscale for next iteration
            # From inference_coz.py:292-300 (recursive mode logic)
            if step > 0:
                w, h = current_image.size
                new_w, new_h = w // self.upscale, h // self.upscale
                
                # Crop center region - inference_coz.py:299
                cropped = current_image.crop(
                    ((w - new_w) // 2, (h - new_h) // 2,
                     (w + new_w) // 2, (h + new_h) // 2)
                )
                
                # Resize back to original size - inference_coz.py:300
                current_image = cropped.resize((w, h), Image.BICUBIC)
            
            # Apply super-resolution
            current_image = self._recursive_zoom_step(current_image, user_prompt)
            
            if return_intermediate:
                results.append(current_image.copy())
        
        return results if return_intermediate else current_image
    
    def process_with_vlm(self, 
                        image: Image.Image,
                        zoom_steps: int = 4,
                        user_prompt: str = "") -> Image.Image:
        """
        Process with VLM prompts (requires additional VLM model).
        This is an advanced mode that requires Qwen VLM model.
        """
        # Lazy load VLM model
        # From inference_coz.py:218-230 - VLM model loading
        if not hasattr(self, 'vlm_model'):
            print("Loading VLM model (this may take a while)...")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info  # inference_coz.py:220
            
            vlm_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"  # inference_coz.py:222
            self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
            self.process_vision_info = process_vision_info
        
        # Process similar to regular process but with VLM prompts
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        current_image = self._resize_and_center_crop(image, self.process_size)
        
        for step in range(zoom_steps):
            if step > 0:
                w, h = current_image.size
                new_w, new_h = w // self.upscale, h // self.upscale
                cropped = current_image.crop(
                    ((w - new_w) // 2, (h - new_h) // 2,
                     (w + new_w) // 2, (h + new_h) // 2)
                )
                current_image = cropped.resize((w, h), Image.BICUBIC)
            
            # Generate VLM prompt
            # From inference_coz.py:46-124 - VLM prompt generation for recursive mode
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                current_image.save(tmp.name)
                tmp_path = tmp.name
            
            # Message setup from inference_coz.py:47-57
            messages = [
                {"role": "system", "content": "What is in this image? Give me a set of words."},
                {"role": "user", "content": [{"type": "image", "image": tmp_path}]}
            ]
            
            # VLM processing from inference_coz.py:58-66
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            
            # Generation from inference_coz.py:116-122
            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Prompt construction from inference_coz.py:124
            prompt_text = f"{output_text}, {user_prompt}," if user_prompt else output_text
            os.unlink(tmp_path)
            
            # Apply SR with VLM prompt
            lq = self.tensor_transforms(current_image).unsqueeze(0).to(self.device)
            sr_output = self._apply_super_resolution(lq, prompt_text)
            current_image = wavelet_color_fix(target=sr_output, source=current_image)
        
        return current_image


# Example usage
if __name__ == "__main__":
    # Initialize service once
    service = ChainOfZoomService()
    
    # Process an image
    input_image = Image.open("samples/0064.png")
    
    # Simple usage with default settings
    output = service.process(input_image)
    output.save("output_default.png")
    
    # With custom zoom steps and prompt
    output = service.process(input_image, zoom_steps=2, user_prompt="highly detailed")
    output.save("output_custom.png")
    
    # Get all intermediate results
    all_steps = service.process(input_image, zoom_steps=3, return_intermediate=True)
    for i, img in enumerate(all_steps):
        img.save(f"output_step_{i}.png")