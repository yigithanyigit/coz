#!/usr/bin/env python3
"""
Simple client to upscale images by any scale factor
"""
import requests
import base64
from PIL import Image
import io
import sys


def upscale(image_path: str, scale: float, output_path: str = None):
    """
    Upscale an image by any scale factor.
    
    Args:
        image_path: Path to input image
        scale: Scale factor (e.g., 2, 3, 4, 5.5, etc.)
        output_path: Where to save (optional)
    """
    # For the standard API (coz_portable_service.py)
    # Note: This will process at 512x512, so aspect ratio won't be preserved
    
    # Calculate zoom steps (each step = 4x)
    import math
    zoom_steps = max(1, min(8, math.ceil(math.log(scale, 4))))
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # Get original size
    img = Image.open(image_path)
    print(f"Original: {img.size[0]}x{img.size[1]}")
    print(f"Target: {int(img.size[0]*scale)}x{int(img.size[1]*scale)} ({scale}x)")
    
    # Call API
    response = requests.post(
        "http://localhost:8000/process",
        json={
            "image": image_base64,
            "zoom_steps": zoom_steps,
            "user_prompt": ""
        }
    )
    
    if response.status_code == 200:
        # Get result
        result = response.json()
        image_data = base64.b64decode(result['image'])
        result_img = Image.open(io.BytesIO(image_data))
        
        # Note: Result will be 512x512 due to current implementation
        print(f"Result: {result_img.size} (fixed processing size)")
        
        # Save
        if not output_path:
            output_path = f"upscaled_{scale}x.png"
        result_img.save(output_path)
        print(f"Saved to: {output_path}")
        
    else:
        print(f"Error: {response.status_code}")


def upscale_flexible(image_path: str, scale: float, output_path: str = None):
    """
    Use the flexible API that preserves aspect ratio.
    Requires running: python coz_flexible_service.py --serve
    """
    # Read and encode image  
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # Call flexible API
    response = requests.post(
        "http://localhost:8000/upscale",
        json={
            "image": image_base64,
            "scale": scale,
            "user_prompt": "highly detailed"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Original: {result['original_size']}")
        print(f"Output: {result['output_size']}")
        
        # Save
        image_data = base64.b64decode(result['image'])
        result_img = Image.open(io.BytesIO(image_data))
        
        if not output_path:
            output_path = f"upscaled_flexible_{scale}x.png"
        result_img.save(output_path)
        print(f"Saved to: {output_path}")
    else:
        print(f"Error: {response.status_code}")
        print("Make sure to run: python coz_flexible_service.py --serve")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_upscale_client.py <image> <scale>")
        print("Example: python simple_upscale_client.py image.png 2.5")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scale = float(sys.argv[2])
    
    # Check which server is running by trying the flexible endpoint first
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200 and "Flexible" in response.text:
            print("\n=== Using flexible API (aspect-ratio preserving) ===")
            upscale_flexible(image_path, scale)
        else:
            print("\n=== Using standard API (512x512 processing) ===")
            upscale(image_path, scale)
    except:
        print("\n=== Trying standard API (512x512 processing) ===")
        upscale(image_path, scale)