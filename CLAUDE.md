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

## Implementation Journey

### Phase 1: Understanding the Repository

**Key Findings:**
- The repository includes pre-trained checkpoint files (SR_LoRA, SR_VAE, DAPE)
- Only the RAM base model (~5.6GB) needs to be downloaded
- The core logic is in `inference_coz.py` with complex file I/O and intermediate saves
- Multiple recursion modes with different prompt generation strategies

**File Structure Analysis:**
- `osediff_sd3.py`: Contains the SR model classes (OSEDiff_SD3_TEST, SD3Euler)
- `ram/`: Complete RAM model implementation with LoRA support
- `lora/`: LoRA layer implementations
- `utils/wavelet_color_fix.py`: Color correction utilities
- `ckpt/`: Checkpoint files (included in repo except RAM base model)

### Phase 2: Creating a Unified Service

**Initial Implementation (`coz_service_unified.py`):**
- Consolidated scattered logic into a single `ChainOfZoomService` class
- Fixed configuration to recommended preset (DAPE + recursive mode)
- Automatic RAM model download on initialization
- Removed file I/O overhead, process in memory

**API Wrapper (`coz_api_minimal.py`):**
- Minimal FastAPI with just `/process` and `/health` endpoints
- Base64 image encoding for REST API
- Fixed 512x512 processing size (limitation)

### Phase 3: Making it Pip-Installable

**Package Structure (`pyproject.toml`):**
- Used modern pyproject.toml with hatchling
- No directory reorganization needed
- Created `serve.py` entry point
- Console script: `coz-serve`

**Issues Encountered:**
- Module import path confusion (fixed by using direct `serve:app`)
- Checkpoint files already in repo (only RAM needs download)

### Phase 4: Portable Self-Contained Service

**`coz_portable_service.py`:**
- Self-contained class with automatic model downloads
- Multiple usage modes: CLI, API server, or library
- Downloads to `~/.cache/chain-of-zoom/`
- Flexible initialization options

**Key Features:**
```python
# As a library
service = ChainOfZoomService()
output = service.process_image("input.png", zoom_steps=4)

# As API server
python coz_portable_service.py --serve

# As CLI tool
python coz_portable_service.py --input img.png --output result.png
```

### Phase 5: Flexible Scaling for Arbitrary Dimensions

**Problem:** Original implementation processes at fixed 512x512, losing aspect ratio

**Solution (`coz_flexible_service.py`):**
- `FlexibleChainOfZoomService` extends base service
- `process_image_with_scale()` method for arbitrary scaling
- Preserves aspect ratio: n×m → (n×s)×(m×s)
- Supports any scale factor (not just powers of 4)

**Implementation Details:**
- Calculate zoom steps: `ceil(log4(scale))`
- Process at original aspect ratio
- Final resize to exact target dimensions
- `max_side` parameter to prevent OOM (default: 8192)

**API Enhancement:**
- New endpoint: `POST /upscale` with scale parameter
- Returns exact scaled dimensions
- Handles non-power-of-4 scales gracefully

### Phase 6: Client Tools

**Created Multiple Clients:**
1. `test_coz_api.py` - Basic testing client
2. `client_example.py` - Shows scale calculations
3. `simple_upscale_client.py` - Auto-detects which server
4. `upscale.py` - Clean CLI for flexible scaling

**Usage Example:**
```bash
# Start flexible server
python coz_flexible_service.py --serve

# Upscale image by 2.5x
python upscale.py image.png 2.5
```

## Technical Implementation Notes

### Model Loading Strategy
- SD3 components distributed across GPUs if available
- Text encoders: fp16 for memory efficiency
- Transformer/VAE: fp32 for stability
- LoRA weights injected at runtime

### Scaling Mathematics
- Each zoom step = 4x upscale
- For scale S: need `ceil(log4(S))` zoom steps
- Example: 2x scale = 1 step, 5x scale = 2 steps (achieves 16x, then downscale)

### Memory Considerations
- Fixed 512x512 processing in original implementation
- Flexible version processes at scaled dimensions
- GPU memory usage increases with zoom steps
- Recommended: 24GB+ VRAM for high resolutions

### Current Limitations
1. Maximum 8 zoom steps (256x theoretical max)
2. Processing time increases linearly with zoom steps
3. Very large images may require tiling (not implemented)
4. Prompt quality affects output quality significantly

## Lessons Learned

1. **Research Code vs Production**: Research code often has complex file I/O and intermediate outputs that aren't needed for production serving

2. **Model Distribution**: Including checkpoint files in the repo (except large base models) simplifies deployment

3. **Flexibility vs Simplicity**: Started with fixed configuration for simplicity, but real-world usage requires flexibility (aspect ratio, arbitrary scales)

4. **API Design**: Base64 encoding works but has size limitations; could use multipart uploads for large images

5. **Automatic Downloads**: Following HuggingFace pattern of automatic model downloads improves user experience

## Future Improvements

1. **Tiling Support**: For images larger than GPU memory
2. **Batch Processing**: Process multiple images efficiently
3. **Streaming**: Progressive output for long processing times
4. **Docker Image**: With pre-downloaded models
5. **WebUI**: User-friendly interface
6. **Optimization**: Implement efficient attention mechanisms for larger images