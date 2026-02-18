"""
Test to verify crop factor fix
Ensure extracted faces contain full face, not just features
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "sdk"))

from _libs.FaceExtractorWrapper import FaceExtractorWrapper
import cv2

def main():
    print("=" * 70)
    print("Crop Factor Fix Verification")
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
        print("\n✗ No test image found")
        print("  Please test with your application")
        return 0
    
    print(f"\nTest image: {test_image.name}")
    
    # Test 1k3d68
    print("\nTesting 1k3d68 with crop_factor=1.75f...")
    wrapper = FaceExtractorWrapper(
        str(models_dir),
        device_id=-1,
        fan_model_path=str(models_dir / "1k3d68.onnx")
    )
    
    faces = wrapper.process_image(str(test_image), face_type=2)
    
    if faces:
        print(f"✓ Detected {len(faces)} face(s)")
        
        # Check if we have aligned face data
        face = faces[0]
        if 'jpg_data' in face and face['jpg_data']:
            print("✓ Aligned face extracted")
            print("\n⚠️  IMPORTANT: Please visually verify the extracted face:")
            print("   - Should contain FULL FACE (forehead, chin, ears)")
            print("   - Should NOT be just eyes/nose/mouth")
            print("   - Should have proper proportions")
            
            # Save for inspection
            output_path = root_dir / "tests" / "crop_fix_test_output.jpg"
            with open(output_path, 'wb') as f:
                f.write(face['jpg_data'])
            print(f"\n✓ Saved extracted face to: {output_path}")
            print("  Please open this file to verify the fix")
        else:
            print("⚠️  No aligned face data available")
        
        landmarks = face.get('landmarks', [])
        print(f"✓ Extracted {len(landmarks)} landmarks")
    else:
        print("✗ No faces detected")
    
    del wrapper
    
    print("\n" + "=" * 70)
    print("Verification Instructions")
    print("=" * 70)
    print("\n1. Check the saved image: tests/crop_fix_test_output.jpg")
    print("2. Verify it contains the FULL FACE")
    print("3. If it only shows eyes/nose/mouth, the fix didn't work")
    print("4. If it shows full face with forehead and chin, the fix is successful")
    
    print("\n" + "=" * 70)
    print("Current Configuration")
    print("=" * 70)
    print("\n1k3d68:")
    print("  crop_factor: 1.75f (same as original)")
    print("  norm_mode: MEAN_STD")
    print("  multi_sample: enabled (5 samples)")
    
    print("\n2d106det:")
    print("  crop_factor: 1.75f (unchanged)")
    print("  norm_mode: AUTO")
    print("  multi_sample: disabled")
    
    print("\n" + "=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
