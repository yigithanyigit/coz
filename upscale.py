#!/usr/bin/env python3
"""
Simple upscaling client for Chain-of-Zoom
Works with coz_flexible_service.py
"""
import requests
import base64
from PIL import Image
import io
import sys
import os


def upscale_image(image_path: str, scale: float, output_path: str = None, api_url: str = "http://localhost:8000"):
    """Upscale an image using the Chain-of-Zoom API."""
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return False
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # Get original dimensions
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    print(f"Input: {image_path} ({orig_w}x{orig_h})")
    print(f"Scale: {scale}x")
    print(f"Target: {int(orig_w * scale)}x{int(orig_h * scale)}")
    
    # Call API
    print("\nProcessing...")
    try:
        response = requests.post(
            f"{api_url}/upscale",
            json={
                "image": image_base64,
                "scale": scale,
                "user_prompt": "highly detailed, sharp"
            },
            timeout=300  # 5 minutes timeout for large images
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Decode result
            image_data = base64.b64decode(result['image'])
            result_img = Image.open(io.BytesIO(image_data))
            
            # Generate output filename if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"{base_name}_x{scale}.png"
            
            # Save
            result_img.save(output_path)
            print(f"\n✓ Success!")
            print(f"Output: {output_path} ({result['output_size'][0]}x{result['output_size'][1]})")
            return True
            
        else:
            print(f"\n✗ Error: {response.status_code}")
            if response.status_code == 404:
                print("Make sure you're running: python coz_flexible_service.py --serve")
            else:
                print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Cannot connect to API server")
        print("Make sure the server is running:")
        print("  python coz_flexible_service.py --serve")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Chain-of-Zoom Upscaler")
        print("=" * 30)
        print("\nUsage: python upscale.py <image> <scale> [output]")
        print("\nExamples:")
        print("  python upscale.py photo.jpg 2")
        print("  python upscale.py photo.jpg 3.5 output.png")
        print("  python upscale.py samples/0064.png 4")
        print("\nNote: Requires server running:")
        print("  python coz_flexible_service.py --serve")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scale = float(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Validate scale
    if scale < 1:
        print("Error: Scale must be >= 1")
        sys.exit(1)
    if scale > 256:
        print("Warning: Very large scale factor. Maximum recommended is 256.")
    
    # Process
    success = upscale_image(image_path, scale, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()