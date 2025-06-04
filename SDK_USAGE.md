# Chain-of-Zoom Python SDK

## Installation

```bash
pip install git+https://github.com/cccntu/Chain-of-Zoom.git
```

The first time you use the service, it will automatically download the required RAM model (~5.6GB) to `~/.cache/chain-of-zoom/`.

## Usage

### As a Python Library

```python
from chain_of_zoom import ChainOfZoomService

# Initialize service
service = ChainOfZoomService()

# Zoom into an image
result = service.zoom(
    "image.jpg",
    zoom_steps=4,      # 256x zoom
    center=(0.3, 0.7), # Focus on specific point
)
result.save("zoomed.png")

# Get all intermediate zoom levels
results = service.zoom(
    "image.jpg",
    zoom_steps=3,
    return_intermediate=True
)
for i, img in enumerate(results):
    img.save(f"zoom_{4**i}x.png")
```

### Running the API Server

```python
# Start the server
import uvicorn
from zoom_api import app

uvicorn.run(app, host="0.0.0.0", port=8000)
```

Or from command line:
```bash
python -m uvicorn zoom_api:app --host 0.0.0.0 --port 8000
```

### Using the API Client

```python
from zoom_client import zoom_into_image

# Simple usage
zoom_into_image("photo.jpg", zoom_steps=4)

# With custom center point
zoom_into_image("photo.jpg", zoom_steps=3, center=(0.3, 0.7))
```

## API Reference

### ChainOfZoomService

Main service class for performing extreme zoom operations.

**Methods:**

- `zoom(image, zoom_steps=4, center=None, user_prompt="", return_intermediate=False)`
  - `image`: PIL Image, file path, or bytes
  - `zoom_steps`: Number of 4x zoom iterations (1-8)
  - `center`: (x, y) tuple in [0, 1] range, default (0.5, 0.5)
  - `user_prompt`: Additional text for prompt generation
  - `return_intermediate`: Return all zoom levels if True

### Zoom Factors

- 1 step = 4x zoom
- 2 steps = 16x zoom  
- 3 steps = 64x zoom
- 4 steps = 256x zoom
- 5 steps = 1,024x zoom
- 6 steps = 4,096x zoom
- 7 steps = 16,384x zoom
- 8 steps = 65,536x zoom