# Testing Guide for Chain-of-Zoom Service

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required model checkpoints:
- Place models in `ckpt/` directory as specified in the README
- Required files:
  - `ckpt/SR_LoRA/model_20001.pkl`
  - `ckpt/SR_VAE/vae_encoder_20001.pt`
  - `ckpt/RAM/ram_swin_large_14m.pth`
  - `ckpt/DAPE/DAPE.pth`

## Testing Methods

### 1. Direct Library Usage Test

```python
from coz_service_unified import ChainOfZoomService
from PIL import Image

# Initialize service
service = ChainOfZoomService()

# Test with a sample image
input_image = Image.open("samples/0064.png")
output = service.process(input_image, zoom_steps=2)
output.save("test_output_direct.png")
print("Test completed! Check test_output_direct.png")
```

### 2. API Server Test

Start the server:
```bash
python coz_api_minimal.py
```

The server will start on `http://localhost:8000`

### 3. Test with Provided Client

In another terminal:
```bash
# Basic test
python test_coz_api.py samples/0064.png

# With custom parameters
python test_coz_api.py samples/0064.png 3 "highly detailed"
```

### 4. Test with cURL

```bash
# First, encode an image to base64
base64 -i samples/0064.png > image.b64

# Make API request
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(cat image.b64)'",
    "zoom_steps": 2,
    "user_prompt": ""
  }' > response.json

# Extract result (requires jq)
jq -r .image response.json | base64 -d > output_curl.png
```

### 5. Test with Python Requests

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
with open("samples/0064.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/process",
    json={
        "image": image_base64,
        "zoom_steps": 2,
        "user_prompt": "highly detailed"
    }
)

if response.status_code == 200:
    result = response.json()
    # Decode and save
    image_data = base64.b64decode(result["image"])
    image = Image.open(io.BytesIO(image_data))
    image.save("output_requests.png")
    print(f"Success! Resolution: {result['final_resolution']}")
else:
    print(f"Error: {response.text}")
```

### 6. API Health Check

```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy", "models_loaded": true}
```

## Test Cases

### Basic Functionality Test
1. Process with default settings (4 zoom steps)
2. Process with 1 zoom step (minimal)
3. Process with 8 zoom steps (maximum)
4. Process with user prompt
5. Process different image formats (PNG, JPEG)

### Error Handling Test
1. Invalid base64 image
2. zoom_steps out of range (0, 9)
3. Corrupted image data

### Performance Test
- Measure processing time for different zoom steps
- Monitor memory usage during processing

## Expected Results

- Each zoom step should produce a 512x512 image
- The image should show progressively more detail with each zoom
- Color consistency should be maintained (wavelet fix)
- Processing time: ~10-30 seconds per zoom step (GPU dependent)

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or use single GPU mode
2. **Model not found**: Ensure all checkpoint files are in correct paths
3. **Import errors**: Check all dependencies are installed
4. **Slow processing**: Ensure GPU is being used (check nvidia-smi)

## Sample Test Script

Save as `run_tests.py`:

```python
import time
import json
from coz_service_unified import ChainOfZoomService
from PIL import Image

def test_direct():
    print("Testing direct service usage...")
    service = ChainOfZoomService()
    
    for img_path in ["samples/0064.png", "samples/0245.png"]:
        print(f"Processing {img_path}...")
        start = time.time()
        
        input_image = Image.open(img_path)
        output = service.process(input_image, zoom_steps=2)
        output.save(f"test_{img_path.split('/')[-1]}")
        
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    test_direct()
```