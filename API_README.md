# Chain-of-Zoom FastAPI Service

A unified API service for Chain-of-Zoom that automatically handles model downloads and provides a simple REST interface.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the service:**
   ```bash
   python coz_api_minimal.py
   ```
   
   On first run, the service will automatically download the RAM model (~5.6GB) from HuggingFace.
   All other required models are already included in the repository.

3. **Test the service:**
   ```bash
   python test_coz_api.py samples/0064.png 4
   ```

## API Endpoints

### POST /process
Process an image through Chain-of-Zoom.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "zoom_steps": 4,
  "user_prompt": "optional text"
}
```

**Response:**
```json
{
  "success": true,
  "image": "base64_encoded_result",
  "zoom_steps": 4,
  "final_resolution": [512, 512]
}
```

### GET /health
Check service status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Python Client Example

```python
import requests
import base64
from PIL import Image
import io

# Encode image
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/process",
    json={
        "image": image_base64,
        "zoom_steps": 4,
        "user_prompt": "highly detailed"
    }
)

# Save result
if response.status_code == 200:
    result = response.json()
    image_data = base64.b64decode(result["image"])
    Image.open(io.BytesIO(image_data)).save("output.png")
```

## Service Configuration

The service uses the recommended Chain-of-Zoom configuration:
- **Model**: Stable Diffusion 3 Medium with LoRA
- **Prompt Generation**: DAPE (fine-tuned RAM)
- **Processing**: Recursive mode with wavelet color correction
- **Upscale Factor**: 4x per zoom step

## Models

All required models are either included in the repository or downloaded automatically:

| Model | Size | Location | Status |
|-------|------|----------|--------|
| SR LoRA | 8.1 MB | `ckpt/SR_LoRA/model_20001.pkl` | ✓ Included |
| VAE Encoder | 69.3 MB | `ckpt/SR_VAE/vae_encoder_20001.pt` | ✓ Included |
| DAPE | 7.2 MB | `ckpt/DAPE/DAPE.pth` | ✓ Included |
| RAM Base | ~5.6 GB | `ckpt/RAM/ram_swin_large_14m.pth` | ⬇️ Auto-download |
| SD3 Medium | ~4.5 GB | HuggingFace cache | ⬇️ Auto-download |

## GPU Requirements

- **Minimum**: 1 GPU with 24GB VRAM
- **Recommended**: 2 GPUs for better performance
- The service automatically distributes models across available GPUs

## Troubleshooting

1. **CUDA out of memory**: Reduce `zoom_steps` or ensure you have sufficient GPU memory
2. **Download failures**: Check your internet connection and HuggingFace access
3. **Slow first run**: Model downloads only happen once and are cached