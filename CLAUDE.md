# Chain-of-Zoom Service Implementation

## Repository Overview

Chain-of-Zoom (CoZ) is a model-agnostic framework for extreme super-resolution that decomposes the task into an autoregressive chain of intermediate scale-states. Instead of training models for higher magnifications, it reuses a 4x SR model recursively with multi-scale-aware prompts.

### Key Components

1. **SR Model**: Stable Diffusion 3 with custom LoRA adaptations
2. **Prompt Generation**: 
   - DAPE (Recognize Anything Model with fine-tuning)
   - VLM (Qwen2.5-VL-3B for advanced prompting)
3. **Color Correction**: Wavelet or AdaIN color fix
4. **Recursion Types**: recursive, onestep, recursive_multiscale

### Core Algorithm

For each zoom step:
1. If not first iteration, crop center region and upscale back to original size
2. Generate text prompt from current image (DAPE/VLM)
3. Apply super-resolution with the prompt
4. Apply color correction to maintain consistency
5. Repeat for desired number of zoom steps

## Implementation Details

### Model Architecture

- **OSEDiff_SD3_TEST**: Main SR model class that loads LoRA weights into SD3
- **SD3Euler**: Base SD3 model wrapper with text encoders, VAE, and transformer
- **LoRA Injection**: Applied to transformer (AdaLayerNormZero layers) and VAE encoder

### Recommended Configuration

Based on the scripts and README example:
- **Prompt Type**: VLM with recursive_multiscale (best) or DAPE with recursive (faster)
- **Models**: SD3-medium, RAM with DAPE fine-tuning
- **Parameters**: 4x upscale, 512px processing, wavelet color fix
- **Zoom Steps**: 4 (default, can go up to 8)

### Memory Optimization

- Multi-GPU: Text encoders on GPU:0, transformer/VAE on GPU:1
- Single GPU: All on same device, optional --efficient_memory flag
- Efficient mode moves models between CPU/GPU as needed

## Service Implementation

Created a unified service with two files:

### 1. coz_service_unified.py

Single class `ChainOfZoomService` that encapsulates all logic:
- Fixed to DAPE+recursive preset (best balance of quality/speed)
- Loads models once in `__init__`
- Main `process()` method handles the recursive zoom
- Clean separation of concerns with private methods
- Optional VLM mode available via `process_with_vlm()`

### 2. coz_api_minimal.py

Minimal FastAPI wrapper:
- `/process` - Returns base64 encoded result
- `/health` - Service health check
- Simple request model: image, zoom_steps, user_prompt

### Key Design Decisions

1. **Single Preset**: Removed configuration complexity, uses recommended DAPE+recursive
2. **In-Memory Processing**: No intermediate file I/O unlike the CLI
3. **Unified Class**: All logic in one place, no scattered functions
4. **Minimal API**: Only essential endpoints, no configuration options

### Usage

```python
# As a library
from coz_service_unified import ChainOfZoomService
service = ChainOfZoomService()
output = service.process(input_image, zoom_steps=4)

# As an API
python coz_api_minimal.py
# Test with: python test_coz_api.py samples/0064.png 4
```

## Technical Notes

- The service maintains the same core algorithm as the CLI but optimized for serving
- Text encoders use fp16, transformer/VAE use fp32 for stability
- LoRA rank is fixed at 4 (optimal for the pre-trained weights)
- Each zoom step processes a 512x512 image regardless of recursion depth
- Color correction is essential for maintaining visual consistency across zoom levels