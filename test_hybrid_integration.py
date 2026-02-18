import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Setup path to include _libs
base_dir = Path(__file__).parent
libs_dir = base_dir / "_libs"
sys.path.insert(0, str(libs_dir))

def main():
    print("Testing Hybrid Architecture Integration...")
    
    try:
        from facelib import S3FDExtractor, FANExtractor
        print("Successfully imported facelib.")
    except ImportError as e:
        print(f"Failed to import facelib: {e}")
        return

    # Initialize Extractors
    try:
        print("Initializing S3FDExtractor...")
        s3fd = S3FDExtractor(place_model_on_cpu=False)
        print("S3FDExtractor initialized.")
        
        print("Initializing FANExtractor...")
        fan = FANExtractor(place_model_on_cpu=False)
        print("FANExtractor initialized.")
    except Exception as e:
        print(f"Failed to initialize extractors: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load test image
    # Use one from data/input if available, or generate a dummy one
    img_path = base_dir.parent / "data" / "input" / "原版提取" / "00001_0.jpg"
    
    # Use unicode aware reading
    try:
        with open(str(img_path), "rb") as f:
            bytes_data = bytearray(f.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Failed to read image at {img_path}: {e}")
        # Try finding another image
        try:
            jpgs = list(base_dir.parent.rglob("*.jpg"))
            if jpgs:
                img_path = jpgs[0]
                print(f"Trying alternative image: {img_path}")
                with open(str(img_path), "rb") as f:
                    bytes_data = bytearray(f.read())
                    numpyarray = np.asarray(bytes_data, dtype=np.uint8)
                    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        except Exception as e2:
            print(f"Failed to find any image: {e2}")
            img = None

    if img is None:
        print("Failed to read image.")
        return

    print(f"Image shape: {img.shape}")

    # Run S3FD
    try:
        print("Running S3FD detection...")
        faces = s3fd.extract(img, is_bgr=True)
        print(f"Detected {len(faces)} faces.")
        for i, face in enumerate(faces):
            print(f"Face {i}: {face}")
    except Exception as e:
        print(f"S3FD detection failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run FAN
    if len(faces) > 0:
        try:
            print("Running FAN extraction...")
            landmarks = fan.extract(img, faces, is_bgr=True)
            print(f"Extracted {len(landmarks)} sets of landmarks.")
            for i, lm in enumerate(landmarks):
                print(f"Landmarks {i} shape: {lm.shape}")
                print(f"Landmarks {i} first 5 points:\n{lm[:5]}")
        except Exception as e:
            print(f"FAN extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("Skipping FAN extraction (no faces).")

    print("Integration test passed!")

if __name__ == "__main__":
    main()
