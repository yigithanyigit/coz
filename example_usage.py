#!/usr/bin/env python3
"""
Example usage of the portable Chain-of-Zoom service
"""
from coz_portable_service import ChainOfZoomService
from PIL import Image

# Example 1: Basic usage
print("Example 1: Basic usage")
service = ChainOfZoomService()
output = service.process_image("samples/0064.png", zoom_steps=2)
output.save("example_output.png")
print("Saved to example_output.png\n")

# Example 2: With custom prompt
print("Example 2: With custom prompt")
output = service.process_image(
    "samples/0064.png", 
    zoom_steps=3,
    user_prompt="highly detailed, sharp"
)
output.save("example_output_prompt.png")
print("Saved to example_output_prompt.png\n")

# Example 3: Process from PIL Image
print("Example 3: From PIL Image")
input_image = Image.open("samples/0064.png")
output = service.process_image(input_image, zoom_steps=2)
output.save("example_output_pil.png")
print("Saved to example_output_pil.png\n")

# Example 4: Custom model directory
print("Example 4: Custom model directory")
custom_service = ChainOfZoomService(
    model_dir="/tmp/coz_models",
    device="cuda"
)
# Use custom_service...