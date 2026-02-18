
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time

# Add current directory to sys.path to ensure imports work
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

try:
    from _libs.FaceExtractorWrapper import FaceExtractorWrapper
except ImportError:
    print("Failed to import FaceExtractorWrapper. Make sure _libs is in the python path.")
    sys.path.append(str(current_dir / "_libs"))
    from FaceExtractorWrapper import FaceExtractorWrapper

def main():
    # Paths
    project_root = current_dir
    models_dir = project_root / "models"
    input_dir = Path(r"d:\Program Files\face_classification\TensorFlow_Extract_Revision\data\input\包含面部图片")
    output_dir = project_root / "output_test"
    
    if not models_dir.exists():
        print(f"Models dir not found: {models_dir}")
        return
        
    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}")
        return

    output_dir.mkdir(exist_ok=True)
    
    print(f"Initializing FaceExtractor with models in {models_dir}...")
    try:
        extractor = FaceExtractorWrapper(models_dir)
        print("Initialization successful.")
    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        return

    # Process images
    image_files = [f for f in input_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    print(f"Found {len(image_files)} images.")
    
    total_faces = 0
    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        try:
            print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
            # Test with default (FULL)
            faces = extractor.process_image(img_path, 2)
            
            print(f"  Found {len(faces)} faces.")
            total_faces += len(faces)
            
            for j, face in enumerate(faces):
                # Save face image
                face_filename = f"{img_path.stem}_face_{j}.jpg"
                save_path = output_dir / face_filename
                
                # face['jpg_data'] is bytes. We need to write it to file.
                # It is already encoded as JPG.
                with open(save_path, "wb") as f:
                    f.write(face['jpg_data'])
                
                # Verify aligned landmarks
                # print(f"    Saved to {save_path}")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nProcessing complete.")
    print(f"Total images: {len(image_files)}")
    print(f"Total faces found: {total_faces}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Average time per image: {elapsed/len(image_files):.2f} seconds")

if __name__ == "__main__":
    main()
