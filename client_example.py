#!/usr/bin/env python3
"""
Example client for Chain-of-Zoom API
"""
import requests
import base64
from PIL import Image
import io
import sys


def upscale_image(image_path: str, scale: int, output_path: str = None):
    """
    Upscale an image by a given scale factor using Chain-of-Zoom API.
    
    Args:
        image_path: Path to input image
        scale: Target scale factor (must be power of 4: 4, 16, 64, etc.)
        output_path: Path to save output (optional)
    """
    # Calculate zoom steps needed
    # Each zoom step does 4x upscale, so for scale S we need log4(S) steps
    import math
    zoom_steps = int(math.log(scale, 4))
    
    if 4**zoom_steps != scale:
        print(f"Warning: Scale {scale} is not a power of 4.")
        print(f"Using {zoom_steps} zoom steps for {4**zoom_steps}x upscale instead.")
        scale = 4**zoom_steps
    
    if zoom_steps < 1:
        zoom_steps = 1
    elif zoom_steps > 8:
        print(f"Warning: Maximum 8 zoom steps supported (256x upscale)")
        zoom_steps = 8
        scale = 256
    
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode()
    
    # Get original dimensions
    original_img = Image.open(image_path)
    orig_w, orig_h = original_img.size
    print(f"Original size: {orig_w}x{orig_h}")
    print(f"Target size: {orig_w*scale}x{orig_h*scale} ({scale}x upscale with {zoom_steps} zoom steps)")
    
    # Make API request
    response = requests.post(
        "http://localhost:8000/process",
        json={
            "image": image_base64,
            "zoom_steps": zoom_steps,
            "user_prompt": "highly detailed, sharp"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode result
        result_data = base64.b64decode(result['image'])
        result_image = Image.open(io.BytesIO(result_data))
        
        # The API returns 512x512, but we need to resize to match the scale
        # This is a limitation of the current implementation
        target_w = orig_w * scale
        target_h = orig_h * scale
        
        print(f"API returned: {result_image.size}")
        print(f"Note: Current implementation processes at fixed 512x512")
        print(f"For true {orig_w}x{orig_h} -> {target_w}x{target_h} scaling,")
        print(f"the service would need modification to preserve aspect ratio")
        
        # Save output
        if not output_path:
            output_path = f"{image_path.rsplit('.', 1)[0]}_x{scale}.png"
        result_image.save(output_path)
        print(f"\nSaved to: {output_path}")
        
        return result_image
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def upscale_with_aspect_ratio(image_path: str, scale: int):
    """
    Example of how to handle arbitrary input sizes.
    Note: This requires modifying the service to accept custom process_size.
    """
    print("\nNote: For arbitrary input/output sizes, the service needs modification:")
    print("1. Accept process_size parameter in API")
    print("2. Modify ChainOfZoomService to use input image size instead of fixed 512")
    print("3. Handle non-square images properly")
    print("\nCurrent implementation always processes at 512x512")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client_example.py <image_path> <scale>")
        print("Example: python client_example.py image.png 4")
        print("\nSupported scales: 4, 16, 64, 256 (powers of 4)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scale = int(sys.argv[2])
    
    # Call the API
    upscale_image(image_path, scale)
    
    # Show limitations
    upscale_with_aspect_ratio(image_path, scale)