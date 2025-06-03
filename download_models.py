#!/usr/bin/env python3
"""
Download required model checkpoints for Chain-of-Zoom
"""
import os
import requests
from tqdm import tqdm
import hashlib


def download_file(url, dest_path, expected_size=None):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"✓ {dest_path} already exists")
        return True
    
    print(f"Downloading {os.path.basename(dest_path)}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Downloaded {dest_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {dest_path}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    """Download all required models"""
    
    print("Chain-of-Zoom Model Downloader")
    print("==============================\n")
    
    # Model URLs and paths
    models = {
        "RAM (Recognize Anything Model)": {
            "url": "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth",
            "path": "ckpt/RAM/ram_swin_large_14m.pth",
            "size": "5.6GB"
        },
        "SD3 Medium": {
            "url": "Hugging Face model - requires authentication",
            "path": "Loaded automatically from HuggingFace",
            "note": "This will be downloaded automatically on first run"
        }
    }
    
    print("Required models:\n")
    for name, info in models.items():
        print(f"{name}:")
        print(f"  Path: {info.get('path', 'N/A')}")
        if 'size' in info:
            print(f"  Size: {info['size']}")
        if 'note' in info:
            print(f"  Note: {info['note']}")
        print()
    
    # Check for user-provided checkpoints
    print("\nChecking for user-provided checkpoints...")
    
    required_files = [
        ("ckpt/SR_LoRA/model_20001.pkl", "SR LoRA weights"),
        ("ckpt/SR_VAE/vae_encoder_20001.pt", "VAE encoder weights"),
        ("ckpt/DAPE/DAPE.pth", "DAPE fine-tuned weights"),
    ]
    
    missing = []
    for path, desc in required_files:
        if os.path.exists(path):
            print(f"✓ Found {desc}: {path}")
        else:
            print(f"✗ Missing {desc}: {path}")
            missing.append((path, desc))
    
    if missing:
        print("\n⚠️  Missing required checkpoint files:")
        print("These files are custom trained models from the paper authors.")
        print("Please obtain them from the paper authors or train your own.\n")
        for path, desc in missing:
            print(f"  - {path} ({desc})")
    
    # Download RAM model
    print("\n" + "="*50)
    print("Downloading RAM base model...")
    print("="*50 + "\n")
    
    ram_url = "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"
    ram_path = "ckpt/RAM/ram_swin_large_14m.pth"
    
    if download_file(ram_url, ram_path):
        print("\n✓ RAM model downloaded successfully!")
    else:
        print("\n✗ Failed to download RAM model")
        print("You can manually download from:")
        print(f"  {ram_url}")
        print(f"And save to: {ram_path}")
    
    print("\n" + "="*50)
    print("Setup Summary")
    print("="*50)
    
    # Final check
    all_files = [
        ("ckpt/SR_LoRA/model_20001.pkl", True),
        ("ckpt/SR_VAE/vae_encoder_20001.pt", True),
        ("ckpt/DAPE/DAPE.pth", True),
        ("ckpt/RAM/ram_swin_large_14m.pth", True),
    ]
    
    ready = True
    for path, required in all_files:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {path}")
        if required and not exists:
            ready = False
    
    if ready:
        print("\n✓ All required models are available!")
        print("You can now run the Chain-of-Zoom service.")
    else:
        print("\n⚠️  Some required models are missing.")
        print("Please obtain the missing checkpoint files before running the service.")
    
    print("\nNote: SD3 model will be downloaded automatically from HuggingFace on first run.")
    print("Make sure you have HuggingFace credentials configured if needed.")


if __name__ == "__main__":
    main()