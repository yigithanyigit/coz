#!/usr/bin/env python3
"""
Simple client for Chain-of-Zoom API
"""
import requests
import base64
from PIL import Image
import io
import sys


def zoom_into_image(image_path: str, zoom_steps: int = 4, center: tuple = (0.5, 0.5), output_path: str = None):
    """
    Zoom into a specific region of an image.
    
    Args:
        image_path: Path to input image
        zoom_steps: Number of zoom iterations (1-8)
        center: (x, y) coordinates in 0-1 range
        output_path: Where to save result
    """
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # Calculate zoom factor
    zoom_factor = 4 ** zoom_steps
    print(f"Zooming {zoom_factor}x into ({center[0]:.2f}, {center[1]:.2f})...")
    
    # Call API
    response = requests.post(
        "http://localhost:8000/zoom",
        json={
            "image": image_base64,
            "zoom_steps": zoom_steps,
            "center_x": center[0],
            "center_y": center[1],
            "user_prompt": ""
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode and save result
        image_data = base64.b64decode(result['image'])
        result_img = Image.open(io.BytesIO(image_data))
        
        if not output_path:
            output_path = f"zoom_{zoom_factor}x.png"
        
        result_img.save(output_path)
        print(f"✓ Saved to: {output_path}")
        print(f"  Zoom factor: {result['zoom_factor']}x")
        print(f"  Center: {result['zoom_center']}")
        
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Chain-of-Zoom Client")
        print("=" * 30)
        print("\nUsage: python zoom_client.py <image> [zoom_steps] [center_x] [center_y]")
        print("\nExamples:")
        print("  python zoom_client.py photo.jpg")
        print("  python zoom_client.py photo.jpg 3")
        print("  python zoom_client.py photo.jpg 4 0.3 0.7")
        print("\nNote: Requires server running:")
        print("  python zoom_api.py")
        sys.exit(1)
    
    image_path = sys.argv[1]
    zoom_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    center_x = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    center_y = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    
    zoom_into_image(image_path, zoom_steps, (center_x, center_y))