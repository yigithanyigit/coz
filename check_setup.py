#!/usr/bin/env python3
"""
Check if Chain-of-Zoom is properly set up
"""
import os
import sys
import torch


def check_file(path, description, required=True):
    """Check if a file exists"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    
    if exists:
        size = os.path.getsize(path)
        size_mb = size / (1024 * 1024)
        print(f"{status} {description:<30} {path} ({size_mb:.1f} MB)")
    else:
        print(f"{status} {description:<30} {path}")
        if required:
            print(f"  ⚠️  This file is required!")
    
    return exists


def main():
    print("Chain-of-Zoom Setup Checker")
    print("===========================\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print()
    
    # Check required files
    print("Checking required model files:")
    print("-" * 80)
    
    required_files = [
        ("ckpt/SR_LoRA/model_20001.pkl", "SR LoRA weights", True),
        ("ckpt/SR_VAE/vae_encoder_20001.pt", "VAE encoder weights", True),
        ("ckpt/DAPE/DAPE.pth", "DAPE fine-tuned weights", True),
        ("ckpt/RAM/ram_swin_large_14m.pth", "RAM base model", True),
    ]
    
    all_present = True
    for path, desc, required in required_files:
        if not check_file(path, desc, required) and required:
            all_present = False
    
    print()
    
    # Check sample images
    print("Checking sample images:")
    print("-" * 80)
    
    sample_images = [
        "samples/0064.png",
        "samples/0245.png",
        "samples/0393.png",
        "samples/0457.png",
        "samples/0479.png",
    ]
    
    samples_found = 0
    for path in sample_images:
        if check_file(path, "Sample image", required=False):
            samples_found += 1
    
    print(f"\nFound {samples_found}/{len(sample_images)} sample images")
    
    # Summary
    print("\n" + "="*80)
    if all_present and samples_found > 0:
        print("✓ Setup looks good! You can run the Chain-of-Zoom service.")
        print("\nTo test:")
        print("  python coz_service_unified.py  # Test direct usage")
        print("  python coz_api_minimal.py      # Start API server")
    else:
        print("✗ Setup incomplete!")
        print("\nMissing files:")
        
        if not os.path.exists("ckpt/RAM/ram_swin_large_14m.pth"):
            print("\n1. Download RAM model:")
            print("   python download_models.py")
        
        missing_custom = []
        for path, desc, _ in required_files[:3]:  # First 3 are custom models
            if not os.path.exists(path):
                missing_custom.append((path, desc))
        
        if missing_custom:
            print("\n2. Obtain custom checkpoint files from the paper authors:")
            for path, desc in missing_custom:
                print(f"   - {path} ({desc})")
            
            print("\n   These are trained models specific to the Chain-of-Zoom paper.")
            print("   Check the paper/repository for download instructions.")
        
        if samples_found == 0:
            print("\n3. No sample images found. Add some test images to samples/")
    
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()