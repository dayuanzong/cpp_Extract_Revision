"""
Test output validation functionality
Verify that landmark validation works correctly
"""

import sys
import os
from pathlib import Path

# Get root directory
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "sdk"))

from _libs.FaceExtractorWrapper import FaceExtractorWrapper
import cv2
import numpy as np

def test_normal_case():
    """Test validation with normal images"""
    print("=" * 70)
    print("Test 1: Normal Case Validation")
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
        return
    
    img = cv2.imread(str(test_image))
    if img is None:
        print("✗ Failed to load image")
        return
    
    print(f"✓ Test image: {test_image.name}")
    
    # Test 1k3d68 with validation
    wrapper = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    faces = wrapper.extract_faces(img)
    
    if faces:
        landmarks = faces[0].get('landmarks', [])
        print(f"✓ Extracted {len(landmarks)} landmarks")
        
        # Check if all landmarks are valid
        valid_count = 0
        for x, y in landmarks[:68]:
            if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                valid_count += 1
        
        print(f"✓ Valid landmarks: {valid_count}/68")
        
        if valid_count == 68:
            print("✓ All landmarks passed validation")
        else:
            print(f"⚠ Some landmarks may be invalid: {68 - valid_count} points")
    else:
        print("✗ No faces detected")
    
    del wrapper
    print()

def test_edge_cases():
    """Test validation with edge cases"""
    print("=" * 70)
    print("Test 2: Edge Case Validation")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    
    # Test with very small image
    print("\nTest 2.1: Very small image (50x50)")
    small_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    wrapper = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    faces = wrapper.extract_faces(small_img)
    if faces:
        print("  ✓ Handled small image (detected face)")
    else:
        print("  ✓ Handled small image (no face detected, as expected)")
    
    # Test with very large image
    print("\nTest 2.2: Large image (2000x2000)")
    large_img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    
    faces = wrapper.extract_faces(large_img)
    if faces:
        print("  ✓ Handled large image (detected face)")
    else:
        print("  ✓ Handled large image (no face detected)")
    
    del wrapper
    print()

def test_validation_stats():
    """Test validation statistics across multiple images"""
    print("=" * 70)
    print("Test 3: Validation Statistics")
    print("=" * 70)
    
    models_dir = root_dir / "assets" / "models"
    test_dir = root_dir / "tests" / "test_images"
    
    if not test_dir.exists():
        print("✗ No test_images directory found, skipping test")
        return
    
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not test_images:
        print("✗ No test images found, skipping test")
        return
    
    print(f"✓ Found {len(test_images)} test images")
    
    wrapper = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    total_faces = 0
    total_valid = 0
    total_invalid = 0
    
    for img_path in test_images[:10]:  # Test first 10 images
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        faces = wrapper.extract_faces(img)
        
        for face in faces:
            total_faces += 1
            landmarks = face.get('landmarks', [])
            
            if len(landmarks) >= 68:
                valid_count = 0
                for x, y in landmarks[:68]:
                    if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                        valid_count += 1
                
                if valid_count == 68:
                    total_valid += 1
                else:
                    total_invalid += 1
    
    print(f"\nResults:")
    print(f"  Total faces: {total_faces}")
    print(f"  Valid: {total_valid}")
    print(f"  Invalid: {total_invalid}")
    
    if total_faces > 0:
        valid_rate = (total_valid / total_faces) * 100
        print(f"  Valid rate: {valid_rate:.1f}%")
        
        if valid_rate >= 95:
            print("✓ Excellent validation rate")
        elif valid_rate >= 80:
            print("⚠ Acceptable validation rate")
        else:
            print("✗ Low validation rate, may need investigation")
    
    del wrapper
    print()

def main():
    print("\n" + "=" * 70)
    print("Output Validation Tests")
    print("=" * 70 + "\n")
    
    try:
        test_normal_case()
        test_edge_cases()
        test_validation_stats()
        
        print("=" * 70)
        print("All validation tests completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
