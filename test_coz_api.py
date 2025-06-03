#!/usr/bin/env python3
"""
Simple test client for Chain-of-Zoom API
"""
import requests
import base64
from PIL import Image
import io
import sys


def test_coz_api(image_path: str, zoom_steps: int = 4, user_prompt: str = ""):
    """Test the Chain-of-Zoom API with an image."""
    
    api_url = "http://localhost:8000"
    
    # Check if service is healthy
    health = requests.get(f"{api_url}/health")
    if health.json()["status"] != "healthy":
        print("Service not ready!")
        return
    
    print(f"Processing {image_path} with {zoom_steps} zoom steps...")
    
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode()
    
    # Make request
    payload = {
        "image": image_base64,
        "zoom_steps": zoom_steps,
        "user_prompt": user_prompt
    }
    
    response = requests.post(f"{api_url}/process", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode and save result
        result_data = base64.b64decode(result['image'])
        result_image = Image.open(io.BytesIO(result_data))
        
        output_path = f"output_zoom{zoom_steps}.png"
        result_image.save(output_path)
        
        print(f"✓ Success! Output saved to {output_path}")
        print(f"  Final resolution: {result['final_resolution']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_coz_api.py <image_path> [zoom_steps] [prompt]")
        print("Example: python test_coz_api.py samples/0064.png 4 'highly detailed'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    zoom_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    user_prompt = sys.argv[3] if len(sys.argv) > 3 else ""
    
    test_coz_api(image_path, zoom_steps, user_prompt)