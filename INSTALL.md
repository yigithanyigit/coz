# Installation Guide

## Quick Install (Recommended)

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/cccntu/Chain-of-Zoom.git
```

Or with uv (faster):

```bash
uv pip install git+https://github.com/cccntu/Chain-of-Zoom.git
```

## Usage

After installation, you can start the API server:

```bash
# Run the server
coz-serve

# Or with custom options
coz-serve --host 0.0.0.0 --port 8080
```

On first run, the server will automatically download the RAM model (~5.6GB) to `~/.cache/chain-of-zoom/`.

## Test the API

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Process image
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

## Development Install

For development with editable install:

```bash
git clone https://github.com/cccntu/Chain-of-Zoom.git
cd Chain-of-Zoom
pip install -e .
```

## Docker

Coming soon: Docker image with pre-downloaded models.

## Requirements

- Python 3.8+
- CUDA-capable GPU (24GB+ VRAM recommended)
- ~10GB disk space for models

## Model Storage

Models are stored in:
- Checkpoint files: Included in package
- RAM model: `~/.cache/chain-of-zoom/` (downloaded on first run)
- SD3 model: HuggingFace cache (downloaded on first run)