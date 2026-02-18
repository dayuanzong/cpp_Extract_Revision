"""
Experiment: Find optimal crop factor for 1k3d68
Test range: 1.4f ~ 2.0f, step 0.1f
Compare with 2DFAN as ground truth
"""

import sys
import os
from pathlib import Path
import json
import time

# Get root directory
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "sdk"))

from _libs.FaceExtractorWrapper import FaceExtractorWrapper
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

def test_crop_factor(crop_factor, test_images, models_dir):
    """Test a specific crop factor"""
    print(f"\nTesting crop_factor = {crop_factor:.1f}")
    
    # Temporarily modify the crop factor in the code
    # Note: This requires recompiling with the new factor
    # For now, we'll document the process
    
    # Load 1k3d68 model
    wrapper_1k3d68 = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    # Load 2DFAN model as ground truth
    wrapper_2dfan = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "FAN" / "2DFAN-4.onnx")
    )
    
    errors = []
    inference_times = []
    
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
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
                # Only compare first 68 points
                error = calculate_error(landmarks_1k3d68[:68], landmarks_2dfan[:68])
                errors.append(error)
                inference_times.append(inference_time)
    
    del wrapper_1k3d68
    del wrapper_2dfan
    
    if not errors:
        return None
    
    # Aggregate results
    result = {
        'crop_factor': crop_factor,
        'mean_error': np.mean([e['mean'] for e in errors]),
        'std_error': np.mean([e['std'] for e in errors]),
        'max_error': np.max([e['max'] for e in errors]),
        'median_error': np.median([e['median'] for e in errors]),
        'avg_inference_time_ms': np.mean(inference_times),
        'num_samples': len(errors)
    }
    
    print(f"  Mean error: {result['mean_error']:.2f} px")
    print(f"  Std error: {result['std_error']:.2f} px")
    print(f"  Max error: {result['max_error']:.2f} px")
    print(f"  Inference time: {result['avg_inference_time_ms']:.1f} ms")
    
    return result

def main():
    print("=" * 70)
    print("Crop Factor Optimization Experiment for 1k3d68")
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
        print("\n  Note: This experiment requires manual code modification.")
        print("  Current implementation uses hardcoded crop_factor in C++ code.")
        print("\n  To run this experiment:")
        print("  1. Modify crop_factor in InsightFaceLandmark.cpp")
        print("  2. Recompile: core/build_cpp.bat")
        print("  3. Run this script")
        print("  4. Repeat for each crop_factor value")
        return 1
    
    print(f"\n✓ Found {len(test_images)} test images")
    print(f"✓ 2DFAN model found")
    
    # Test range: 1.4 ~ 2.0, step 0.1
    crop_factors = [round(1.4 + i * 0.1, 1) for i in range(7)]  # 1.4, 1.5, ..., 2.0
    
    print(f"\nCrop factors to test: {crop_factors}")
    print("\n" + "=" * 70)
    print("IMPORTANT: Manual Process Required")
    print("=" * 70)
    print("\nThis experiment requires manual modification of the C++ code:")
    print("1. Open core/src/InsightFaceLandmark.cpp")
    print("2. Find the line: configs[ModelType::MODEL_1K3D68] = {")
    print("3. Modify crop_factor value")
    print("4. Run: core\\build_cpp.bat")
    print("5. Run this script to test")
    print("6. Record results")
    print("7. Repeat for each crop_factor value")
    
    print("\n" + "=" * 70)
    print("Testing Current Configuration")
    print("=" * 70)
    
    # Test current configuration
    current_result = test_crop_factor(1.6, test_images[:5], models_dir)  # Test with first 5 images
    
    if current_result:
        # Save result
        results_file = root_dir / "tests" / "crop_factor_results.json"
        results = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        
        # Check if this crop_factor already exists
        existing = [r for r in results if abs(r['crop_factor'] - current_result['crop_factor']) < 0.01]
        if not existing:
            results.append(current_result)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {results_file}")
        else:
            print(f"\n  Results for crop_factor={current_result['crop_factor']} already exist")
        
        # Show summary if we have multiple results
        if len(results) > 1:
            print("\n" + "=" * 70)
            print("Summary of All Results")
            print("=" * 70)
            results_sorted = sorted(results, key=lambda x: x['mean_error'])
            print(f"\n{'Crop Factor':<12} {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12} {'Time (ms)':<12}")
            print("-" * 70)
            for r in results_sorted:
                print(f"{r['crop_factor']:<12.1f} {r['mean_error']:<12.2f} {r['std_error']:<12.2f} "
                      f"{r['max_error']:<12.2f} {r['avg_inference_time_ms']:<12.1f}")
            
            best = results_sorted[0]
            print(f"\n✓ Best crop_factor: {best['crop_factor']:.1f} (mean error: {best['mean_error']:.2f} px)")
    
    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
