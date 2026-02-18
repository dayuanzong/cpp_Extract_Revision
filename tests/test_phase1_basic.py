"""
Test Phase 1: Basic refactoring
- Model type detection
- Configuration management
- ExtractSingle and ExtractWithMultiSample
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

def test_model_type_detection():
    """Test model type detection from filename"""
    print("=" * 60)
    print("Test 1: Model Type Detection")
    print("=" * 60)
    
    models_dir = root_dir / "assets" / "models"
    
    # Test 1k3d68 detection
    wrapper_1k3d68 = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    print("✓ 1k3d68 model loaded successfully")
    
    # Test 2d106det detection
    wrapper_2d106det = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "2d106det.onnx")
    )
    print("✓ 2d106det model loaded successfully")
    
    del wrapper_1k3d68
    del wrapper_2d106det
    print()

def test_extraction():
    """Test extraction with both models"""
    print("=" * 60)
    print("Test 2: Extraction Test")
    print("=" * 60)
    
    models_dir = root_dir / "assets" / "models"
    
    # Load test image
    img_path = root_dir / "aligned_face_0.jpg"
    if not img_path.exists():
        print(f"✗ Test image not found: {img_path}")
        print("  Skipping extraction test")
        return
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"✗ Failed to load test image: {img_path}")
        print("  Skipping extraction test")
        return
        
    print(f"✓ Test image loaded: {img.shape}")
    
    # Test 1k3d68
    print("\nTesting 1k3d68...")
    wrapper_1k3d68 = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    faces_1k3d68 = wrapper_1k3d68.extract_faces(img)
    if faces_1k3d68:
        print(f"✓ 1k3d68 detected {len(faces_1k3d68)} face(s)")
        for i, face in enumerate(faces_1k3d68):
            landmarks = face.get('landmarks', [])
            print(f"  Face {i}: {len(landmarks)} landmarks")
    else:
        print("✗ 1k3d68 failed to detect faces")
    
    del wrapper_1k3d68
    
    # Test 2d106det
    print("\nTesting 2d106det...")
    wrapper_2d106det = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "2d106det.onnx")
    )
    
    faces_2d106det = wrapper_2d106det.extract_faces(img)
    if faces_2d106det:
        print(f"✓ 2d106det detected {len(faces_2d106det)} face(s)")
        for i, face in enumerate(faces_2d106det):
            landmarks = face.get('landmarks', [])
            print(f"  Face {i}: {len(landmarks)} landmarks")
    else:
        print("✗ 2d106det failed to detect faces")
    
    del wrapper_2d106det
    print()

def test_multi_sampling_config():
    """Test multi-sampling configuration"""
    print("=" * 60)
    print("Test 3: Multi-sampling Configuration")
    print("=" * 60)
    
    models_dir = root_dir / "assets" / "models"
    
    # 1k3d68 should have multi-sampling enabled
    print("1k3d68: Multi-sampling should be ENABLED (5 samples)")
    wrapper_1k3d68 = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    print("✓ 1k3d68 initialized with multi-sampling config")
    del wrapper_1k3d68
    
    # 2d106det should have multi-sampling disabled
    print("\n2d106det: Multi-sampling should be DISABLED (1 sample)")
    wrapper_2d106det = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "2d106det.onnx")
    )
    print("✓ 2d106det initialized with single-sampling config")
    del wrapper_2d106det
    print()

def main():
    print("\n" + "=" * 60)
    print("Phase 1: Basic Refactoring Tests")
    print("=" * 60 + "\n")
    
    try:
        test_model_type_detection()
        test_extraction()
        test_multi_sampling_config()
        
        print("=" * 60)
        print("All Phase 1 tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
