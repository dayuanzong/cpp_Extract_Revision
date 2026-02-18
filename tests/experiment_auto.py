"""
Automated Experiment: Test different configurations for 1k3d68
This script modifies the C++ code, recompiles, and tests automatically
"""

import sys
import os
from pathlib import Path
import json
import time
import subprocess
import re

# Get root directory
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "sdk"))

import cv2
import numpy as np

def calculate_error(landmarks1, landmarks2):
    """Calculate average Euclidean distance between two landmark sets"""
    if len(landmarks1) != len(landmarks2):
        return float('inf')
    
    distances = []
    for p1, p2 in zip(landmarks1, landmarks2):
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        distances.append(dist)
    
    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'max': np.max(distances),
        'median': np.median(distances)
    }

def modify_cpp_config(crop_factor, norm_mode):
    """Modify the C++ configuration"""
    cpp_file = root_dir / "core" / "src" / "InsightFaceLandmark.cpp"
    
    with open(cpp_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    backup_file = cpp_file.with_suffix('.cpp.bak')
    if not backup_file.exists():
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Modify crop_factor for 1k3d68
    pattern = r'(configs\[ModelType::MODEL_1K3D68\]\s*=\s*\{[^}]*?{\s*)(\d+\.?\d*)f(\s*,\s*//\s*crop_factor)'
    replacement = rf'\g<1>{crop_factor}f\g<3>'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Modify norm_mode for 1k3d68
    norm_mode_str = f"NormMode::{norm_mode}"
    pattern = r'(configs\[ModelType::MODEL_1K3D68\]\s*=\s*\{[^}]*?{\s*)(NormMode::\w+)(\s*,\s*//\s*norm_mode)'
    replacement = rf'\g<1>{norm_mode_str}\g<3>'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(cpp_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def restore_cpp_config():
    """Restore original C++ configuration"""
    cpp_file = root_dir / "core" / "src" / "InsightFaceLandmark.cpp"
    backup_file = cpp_file.with_suffix('.cpp.bak')
    
    if backup_file.exists():
        with open(backup_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(cpp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def compile_cpp():
    """Compile the C++ code"""
    build_script = root_dir / "core" / "build_cpp.bat"
    
    print("  Compiling C++ code...")
    try:
        result = subprocess.run(
            [str(build_script)],
            cwd=str(root_dir / "core"),
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print("  ✓ Compilation successful")
            return True
        else:
            print(f"  ✗ Compilation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Compilation error: {e}")
        return False

def test_configuration(crop_factor, norm_mode, test_images, models_dir):
    """Test a specific configuration"""
    print(f"\nTesting: crop_factor={crop_factor}, norm_mode={norm_mode}")
    
    # Modify configuration
    if not modify_cpp_config(crop_factor, norm_mode):
        print("  ✗ Failed to modify configuration")
        return None
    
    # Compile
    if not compile_cpp():
        print("  ✗ Failed to compile")
        restore_cpp_config()
        return None
    
    # Import wrapper after compilation
    from _libs.FaceExtractorWrapper import FaceExtractorWrapper
    
    # Load models
    try:
        wrapper_1k3d68 = FaceExtractorWrapper(
            str(models_dir),
            device_id=-1,
            fan_model_path=str(models_dir / "1k3d68.onnx")
        )
        
        wrapper_2dfan = FaceExtractorWrapper(
            str(models_dir),
            device_id=-1,
            fan_model_path=str(models_dir / "FAN" / "2DFAN-4.onnx")
        )
    except Exception as e:
        print(f"  ✗ Failed to load models: {e}")
        restore_cpp_config()
        return None
    
    errors = []
    inference_times = []
    
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        try:
            # Extract with 1k3d68
            start_time = time.time()
            faces_1k3d68 = wrapper_1k3d68.extract_faces(img)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Extract with 2DFAN
            faces_2dfan = wrapper_2dfan.extract_faces(img)
            
            if faces_1k3d68 and faces_2dfan:
                landmarks_1k3d68 = faces_1k3d68[0].get('landmarks', [])
                landmarks_2dfan = faces_2dfan[0].get('landmarks', [])
                
                if len(landmarks_1k3d68) >= 68 and len(landmarks_2dfan) >= 68:
                    error = calculate_error(landmarks_1k3d68[:68], landmarks_2dfan[:68])
                    errors.append(error)
                    inference_times.append(inference_time)
        except Exception as e:
            print(f"  Warning: Error processing {img_path.name}: {e}")
            continue
    
    del wrapper_1k3d68
    del wrapper_2dfan
    
    if not errors:
        print("  ✗ No valid results")
        return None
    
    # Aggregate results
    result = {
        'crop_factor': crop_factor,
        'norm_mode': norm_mode,
        'mean_error': np.mean([e['mean'] for e in errors]),
        'std_error': np.mean([e['std'] for e in errors]),
        'max_error': np.max([e['max'] for e in errors]),
        'median_error': np.median([e['median'] for e in errors]),
        'avg_inference_time_ms': np.mean(inference_times),
        'num_samples': len(errors)
    }
    
    print(f"  ✓ Mean error: {result['mean_error']:.2f} px")
    print(f"  ✓ Std error: {result['std_error']:.2f} px")
    print(f"  ✓ Inference time: {result['avg_inference_time_ms']:.1f} ms")
    
    return result

def main():
    print("=" * 70)
    print("Automated Configuration Optimization for 1k3d68")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Check if 2DFAN model exists
    fan_model_path = models_dir / "FAN" / "2DFAN-4.onnx"
    if not fan_model_path.exists():
        print(f"\n✗ 2DFAN model not found at: {fan_model_path}")
        print("  Please ensure 2DFAN-4.onnx is in assets/models/FAN/")
        return 1
    
    # Collect test images
    test_images = []
    test_dir = root_dir / "tests" / "test_images"
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not test_images:
        print("\n✗ No test images found in tests/test_images/")
        print("  Please add test images to tests/test_images/")
        return 1
    
    print(f"\n✓ Found {len(test_images)} test images")
    print(f"✓ 2DFAN model found")
    
    # Define test configurations
    crop_factors = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    norm_modes = ["ZERO_ONE", "MEAN_STD"]
    
    print(f"\nCrop factors to test: {crop_factors}")
    print(f"Norm modes to test: {norm_modes}")
    print(f"Total configurations: {len(crop_factors) * len(norm_modes)}")
    
    # Confirm
    print("\n" + "=" * 70)
    print("WARNING: This will modify and recompile C++ code multiple times")
    print("=" * 70)
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted.")
        return 0
    
    # Run experiments
    results = []
    total = len(crop_factors) * len(norm_modes)
    current = 0
    
    try:
        for crop_factor in crop_factors:
            for norm_mode in norm_modes:
                current += 1
                print(f"\n{'='*70}")
                print(f"Progress: {current}/{total}")
                print(f"{'='*70}")
                
                result = test_configuration(
                    crop_factor, 
                    norm_mode, 
                    test_images[:10],  # Use first 10 images
                    models_dir
                )
                
                if result:
                    results.append(result)
                
                # Small delay between tests
                time.sleep(1)
    
    finally:
        # Always restore original configuration
        print("\n" + "=" * 70)
        print("Restoring original configuration...")
        print("=" * 70)
        if restore_cpp_config():
            compile_cpp()
            print("✓ Original configuration restored")
        else:
            print("✗ Failed to restore original configuration")
    
    # Save results
    if results:
        results_file = root_dir / "tests" / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
        
        # Show summary
        print("\n" + "=" * 70)
        print("Summary of All Results")
        print("=" * 70)
        results_sorted = sorted(results, key=lambda x: x['mean_error'])
        print(f"\n{'Crop':<6} {'Norm Mode':<12} {'Mean Err':<10} {'Std Err':<10} {'Max Err':<10} {'Time(ms)':<10}")
        print("-" * 70)
        for r in results_sorted:
            print(f"{r['crop_factor']:<6.1f} {r['norm_mode']:<12} {r['mean_error']:<10.2f} "
                  f"{r['std_error']:<10.2f} {r['max_error']:<10.2f} {r['avg_inference_time_ms']:<10.1f}")
        
        best = results_sorted[0]
        print(f"\n{'='*70}")
        print(f"✓ Best configuration:")
        print(f"  Crop factor: {best['crop_factor']}")
        print(f"  Norm mode: {best['norm_mode']}")
        print(f"  Mean error: {best['mean_error']:.2f} px")
        print(f"  Inference time: {best['avg_inference_time_ms']:.1f} ms")
        print(f"{'='*70}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("Restoring original configuration...")
        restore_cpp_config()
        sys.exit(1)
