#!/usr/bin/env python3
"""
Comprehensive test suite for Chain-of-Zoom service
"""
import os
import sys
import time
import json
import base64
import requests
from PIL import Image
import io
from typing import List, Tuple


def test_direct_service():
    """Test the service directly without API"""
    print("\n=== Testing Direct Service Usage ===")
    
    try:
        from coz_service_unified import ChainOfZoomService
        
        print("Initializing service...")
        start = time.time()
        service = ChainOfZoomService()
        init_time = time.time() - start
        print(f"Service initialized in {init_time:.2f} seconds")
        
        # Test with different zoom steps
        test_cases = [
            ("samples/0064.png", 1, ""),
            ("samples/0064.png", 2, "highly detailed"),
            ("samples/0245.png", 3, ""),
        ]
        
        for img_path, zoom_steps, prompt in test_cases:
            if os.path.exists(img_path):
                print(f"\nProcessing {img_path} with {zoom_steps} zoom steps...")
                start = time.time()
                
                input_image = Image.open(img_path)
                output = service.process(
                    input_image, 
                    zoom_steps=zoom_steps,
                    user_prompt=prompt
                )
                
                output_path = f"test_direct_{os.path.basename(img_path).replace('.png', f'_z{zoom_steps}.png')}"
                output.save(output_path)
                
                elapsed = time.time() - start
                print(f"✓ Saved to {output_path} ({elapsed:.2f}s)")
                print(f"  Input size: {input_image.size}, Output size: {output.size}")
            else:
                print(f"✗ Skipping {img_path} - file not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Direct service test failed: {e}")
        return False


def test_api_health(base_url: str = "http://localhost:8000") -> bool:
    """Test API health endpoint"""
    print("\n=== Testing API Health ===")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(f"✓ Models Loaded: {data['models_loaded']}")
            return data['status'] == 'healthy'
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure server is running:")
        print("  python coz_api_minimal.py")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_api_process(base_url: str = "http://localhost:8000"):
    """Test API processing endpoint"""
    print("\n=== Testing API Processing ===")
    
    test_cases = [
        ("samples/0064.png", 2, ""),
        ("samples/0245.png", 1, "detailed, high quality"),
    ]
    
    for img_path, zoom_steps, prompt in test_cases:
        if not os.path.exists(img_path):
            print(f"✗ Skipping {img_path} - file not found")
            continue
            
        print(f"\nProcessing {img_path} via API...")
        
        try:
            # Encode image
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            
            # Make request
            start = time.time()
            response = requests.post(
                f"{base_url}/process",
                json={
                    "image": image_base64,
                    "zoom_steps": zoom_steps,
                    "user_prompt": prompt
                },
                timeout=120  # 2 minutes timeout
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                
                # Decode and save result
                image_data = base64.b64decode(result["image"])
                output_image = Image.open(io.BytesIO(image_data))
                
                output_path = f"test_api_{os.path.basename(img_path).replace('.png', f'_z{zoom_steps}.png')}"
                output_image.save(output_path)
                
                print(f"✓ Saved to {output_path} ({elapsed:.2f}s)")
                print(f"  Final resolution: {result['final_resolution']}")
            else:
                print(f"✗ API request failed: {response.status_code}")
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"✗ API test error: {e}")


def test_error_handling(base_url: str = "http://localhost:8000"):
    """Test API error handling"""
    print("\n=== Testing Error Handling ===")
    
    # Test invalid base64
    print("\nTesting invalid base64...")
    response = requests.post(
        f"{base_url}/process",
        json={
            "image": "invalid_base64",
            "zoom_steps": 2
        }
    )
    print(f"Invalid base64 response: {response.status_code}")
    if response.status_code >= 400:
        print("✓ Correctly rejected invalid base64")
    
    # Test invalid zoom_steps
    print("\nTesting invalid zoom_steps...")
    with open("samples/0064.png", "rb") as f:
        valid_base64 = base64.b64encode(f.read()).decode()
    
    for invalid_steps in [0, 9, -1]:
        response = requests.post(
            f"{base_url}/process",
            json={
                "image": valid_base64,
                "zoom_steps": invalid_steps
            }
        )
        if response.status_code == 400:
            print(f"✓ Correctly rejected zoom_steps={invalid_steps}")


def benchmark_performance():
    """Benchmark processing performance"""
    print("\n=== Performance Benchmark ===")
    
    try:
        from coz_service_unified import ChainOfZoomService
        
        service = ChainOfZoomService()
        input_image = Image.open("samples/0064.png")
        
        print("\nBenchmarking different zoom steps:")
        print("Zoom Steps | Time (s) | Time per Step (s)")
        print("-" * 40)
        
        for zoom_steps in [1, 2, 3, 4]:
            start = time.time()
            output = service.process(input_image, zoom_steps=zoom_steps)
            elapsed = time.time() - start
            per_step = elapsed / zoom_steps
            
            print(f"    {zoom_steps}      | {elapsed:8.2f} | {per_step:8.2f}")
            
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")


def main():
    """Run all tests"""
    print("Chain-of-Zoom Service Test Suite")
    print("================================")
    
    # Check if running API tests
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        print("\nRunning API tests...")
        if test_api_health():
            test_api_process()
            test_error_handling()
        else:
            print("\n⚠️  API server not running. Start with: python coz_api_minimal.py")
    
    # Run direct tests
    elif len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_performance()
    
    else:
        print("\nUsage:")
        print("  python run_tests.py          # Run direct service tests")
        print("  python run_tests.py --api    # Run API tests (requires server)")
        print("  python run_tests.py --benchmark  # Run performance benchmark")
        print("\nRunning direct service tests...")
        test_direct_service()


if __name__ == "__main__":
    main()