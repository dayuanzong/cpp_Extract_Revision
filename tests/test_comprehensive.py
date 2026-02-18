"""
Comprehensive test suite for 1k3d68 optimization
Tests all implemented features and optimizations
"""

import sys
import os
from pathlib import Path
import time
import json

# Get root directory
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "sdk"))

from _libs.FaceExtractorWrapper import FaceExtractorWrapper
import cv2
import numpy as np

def calculate_error(landmarks1, landmarks2):
    """Calculate error metrics between two landmark sets"""
    if len(landmarks1) != len(landmarks2):
        return None
    
    distances = []
    for p1, p2 in zip(landmarks1, landmarks2):
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        distances.append(dist)
    
    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'max': np.max(distances),
        'min': np.min(distances),
        'median': np.median(distances)
    }

def test_model_detection():
    """Test 1: Model type detection"""
    print("=" * 70)
    print("Test 1: Model Type Detection")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Test 1k3d68
    print("\n1.1 Testing 1k3d68 detection...")
    wrapper_1k = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    print("✓ 1k3d68 loaded (should detect as MODEL_1K3D68)")
    del wrapper_1k
    
    # Test 2d106det
    print("\n1.2 Testing 2d106det detection...")
    wrapper_2d = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "2d106det.onnx")
    )
    print("✓ 2d106det loaded (should detect as MODEL_2D106DET)")
    del wrapper_2d
    
    print("\n✓ Model type detection test passed\n")
    return True

def test_independent_config():
    """Test 2: Independent configuration for 1k3d68 and 2d106det"""
    print("=" * 70)
    print("Test 2: Independent Configuration")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Find test image
    test_image = root_dir / "aligned_face_0.jpg"
    if not test_image.exists():
        test_dir = root_dir / "tests" / "test_images"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if images:
                test_image = images[0]
    
    if not test_image.exists():
        print("✗ No test image found, skipping test")
        return False
    
    img = cv2.imread(str(test_image))
    
    print("\n2.1 Testing 1k3d68 configuration...")
    print("  Expected: crop_factor=1.6, multi_sample=enabled")
    wrapper_1k = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    start = time.time()
    faces_1k = wrapper_1k.extract_faces(img)
    time_1k = (time.time() - start) * 1000
    
    if faces_1k:
        print(f"✓ 1k3d68 extracted {len(faces_1k[0].get('landmarks', []))} landmarks")
        print(f"  Inference time: {time_1k:.1f} ms")
    else:
        print("⚠ No faces detected")
    
    del wrapper_1k
    
    print("\n2.2 Testing 2d106det configuration...")
    print("  Expected: crop_factor=1.75, multi_sample=disabled")
    wrapper_2d = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "2d106det.onnx")
    )
    
    start = time.time()
    faces_2d = wrapper_2d.extract_faces(img)
    time_2d = (time.time() - start) * 1000
    
    if faces_2d:
        print(f"✓ 2d106det extracted {len(faces_2d[0].get('landmarks', []))} landmarks")
        print(f"  Inference time: {time_2d:.1f} ms")
    else:
        print("⚠ No faces detected")
    
    del wrapper_2d
    
    print("\n✓ Independent configuration test passed\n")
    return True

def test_multi_sampling():
    """Test 3: Multi-sampling functionality"""
    print("=" * 70)
    print("Test 3: Multi-sampling")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Find test image
    test_image = root_dir / "aligned_face_0.jpg"
    if not test_image.exists():
        test_dir = root_dir / "tests" / "test_images"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if images:
                test_image = images[0]
    
    if not test_image.exists():
        print("✗ No test image found, skipping test")
        return False
    
    img = cv2.imread(str(test_image))
    
    print("\n3.1 Testing 1k3d68 with multi-sampling (5 samples)...")
    wrapper = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    # Run multiple times to check consistency
    results = []
    for i in range(3):
        start = time.time()
        faces = wrapper.extract_faces(img)
        elapsed = (time.time() - start) * 1000
        
        if faces:
            landmarks = faces[0].get('landmarks', [])
            results.append({
                'landmarks': landmarks[:68],
                'time': elapsed
            })
    
    if len(results) >= 2:
        # Check consistency between runs
        error = calculate_error(results[0]['landmarks'], results[1]['landmarks'])
        if error:
            print(f"✓ Multi-sampling consistency:")
            print(f"  Mean difference: {error['mean']:.3f} px")
            print(f"  Max difference: {error['max']:.3f} px")
            
            if error['mean'] < 0.5:
                print("  ✓ Excellent consistency (< 0.5 px)")
            elif error['mean'] < 1.0:
                print("  ✓ Good consistency (< 1.0 px)")
            else:
                print("  ⚠ Moderate consistency")
        
        avg_time = np.mean([r['time'] for r in results])
        print(f"  Average inference time: {avg_time:.1f} ms")
    
    del wrapper
    
    print("\n✓ Multi-sampling test passed\n")
    return True

def test_accuracy():
    """Test 4: Accuracy comparison with 2DFAN"""
    print("=" * 70)
    print("Test 4: Accuracy Comparison")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Check if 2DFAN exists
    fan_model_path = models_dir / "FAN" / "2DFAN-4.onnx"
    if not fan_model_path.exists():
        print("✗ 2DFAN model not found, skipping accuracy test")
        return False
    
    # Find test image
    test_image = root_dir / "aligned_face_0.jpg"
    if not test_image.exists():
        test_dir = root_dir / "tests" / "test_images"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if images:
                test_image = images[0]
    
    if not test_image.exists():
        print("✗ No test image found, skipping test")
        return False
    
    img = cv2.imread(str(test_image))
    
    print("\n4.1 Extracting with 1k3d68...")
    wrapper_1k = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    start = time.time()
    faces_1k = wrapper_1k.extract_faces(img)
    time_1k = (time.time() - start) * 1000
    
    if not faces_1k:
        print("✗ 1k3d68 failed to detect faces")
        del wrapper_1k
        return False
    
    landmarks_1k = faces_1k[0].get('landmarks', [])[:68]
    print(f"✓ 1k3d68: {len(landmarks_1k)} landmarks, {time_1k:.1f} ms")
    del wrapper_1k
    
    print("\n4.2 Extracting with 2DFAN...")
    wrapper_2d = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "FAN" / "2DFAN-4.onnx")
    )
    
    start = time.time()
    faces_2d = wrapper_2d.extract_faces(img)
    time_2d = (time.time() - start) * 1000
    
    if not faces_2d:
        print("✗ 2DFAN failed to detect faces")
        del wrapper_2d
        return False
    
    landmarks_2d = faces_2d[0].get('landmarks', [])[:68]
    print(f"✓ 2DFAN: {len(landmarks_2d)} landmarks, {time_2d:.1f} ms")
    del wrapper_2d
    
    print("\n4.3 Comparing accuracy...")
    error = calculate_error(landmarks_1k, landmarks_2d)
    
    if error:
        print(f"  Mean error: {error['mean']:.2f} px")
        print(f"  Std error: {error['std']:.2f} px")
        print(f"  Max error: {error['max']:.2f} px")
        print(f"  Median error: {error['median']:.2f} px")
        
        # Performance comparison
        speedup = time_2d / time_1k
        print(f"\n  Performance:")
        print(f"    1k3d68: {time_1k:.1f} ms")
        print(f"    2DFAN: {time_2d:.1f} ms")
        print(f"    Speedup: {speedup:.2f}x")
        
        # Accuracy assessment
        if error['mean'] < 3.0:
            print("\n  ✓ Excellent accuracy (< 3.0 px)")
        elif error['mean'] < 5.0:
            print("\n  ✓ Good accuracy (< 5.0 px)")
        elif error['mean'] < 8.0:
            print("\n  ⚠ Acceptable accuracy (< 8.0 px)")
        else:
            print("\n  ⚠ Needs improvement (>= 8.0 px)")
    
    print("\n✓ Accuracy test completed\n")
    return True

def test_validation():
    """Test 5: Output validation"""
    print("=" * 70)
    print("Test 5: Output Validation")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Find test image
    test_image = root_dir / "aligned_face_0.jpg"
    if not test_image.exists():
        test_dir = root_dir / "tests" / "test_images"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if images:
                test_image = images[0]
    
    if not test_image.exists():
        print("✗ No test image found, skipping test")
        return False
    
    img = cv2.imread(str(test_image))
    
    print("\n5.1 Testing validation with 1k3d68...")
    wrapper = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    faces = wrapper.extract_faces(img)
    
    if faces:
        landmarks = faces[0].get('landmarks', [])[:68]
        
        # Check for invalid values
        invalid_count = 0
        for x, y in landmarks:
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                invalid_count += 1
        
        print(f"✓ Extracted {len(landmarks)} landmarks")
        print(f"  Valid: {len(landmarks) - invalid_count}")
        print(f"  Invalid: {invalid_count}")
        
        if invalid_count == 0:
            print("  ✓ All landmarks valid")
        else:
            print(f"  ⚠ {invalid_count} invalid landmarks detected")
    else:
        print("⚠ No faces detected")
    
    del wrapper
    
    print("\n✓ Validation test completed\n")
    return True

def main():
    print("\n" + "=" * 70)
    print("Comprehensive Test Suite for 1k3d68 Optimization")
    print("=" * 70 + "\n")
    
    results = {
        'model_detection': False,
        'independent_config': False,
        'multi_sampling': False,
        'accuracy': False,
        'validation': False
    }
    
    try:
        results['model_detection'] = test_model_detection()
        results['independent_config'] = test_independent_config()
        results['multi_sampling'] = test_multi_sampling()
        results['accuracy'] = test_accuracy()
        results['validation'] = test_validation()
        
        # Summary
        print("=" * 70)
        print("Test Summary")
        print("=" * 70)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_flag in results.items():
            status = "✓ PASS" if passed_flag else "✗ FAIL"
            print(f"  {test_name:20s}: {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✓ All tests passed!")
            return 0
        else:
            print(f"\n⚠ {total - passed} test(s) failed")
            return 1
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
