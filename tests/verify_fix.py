import sys
import os
import ctypes
import numpy as np
import cv2
from pathlib import Path

# Setup paths
# Script is in tests directory, root is parent
root_dir = Path(__file__).parent.parent.resolve()
dll_path = root_dir / "bin/FaceExtractorDLL.dll"

# Structures
class FaceInfo(ctypes.Structure):
    _fields_ = [
        ("jpg_data", ctypes.POINTER(ctypes.c_ubyte)),
        ("jpg_size", ctypes.c_int),
        ("landmarks", ctypes.c_float * 136),
        ("aligned_landmarks", ctypes.c_float * 136),
        ("source_rect", ctypes.c_float * 4),
        ("detect_score", ctypes.c_float),
        ("blur_variance", ctypes.c_float),
        ("pose_tag", ctypes.c_char * 32),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("blur_class", ctypes.c_int),
        ("mouth_value", ctypes.c_float),
        ("mouth_open", ctypes.c_bool),
        ("valid", ctypes.c_bool),
    ]

def load_lib():
    if not dll_path.exists():
        print(f"DLL not found at {dll_path}")
        return None
    
    # Unified DLL path: all DLLs are in bin directory
    bin_dir = root_dir / "bin"
    
    if hasattr(os, "add_dll_directory"):
        if bin_dir.exists():
            os.add_dll_directory(str(bin_dir))
    
    os.environ["PATH"] = str(bin_dir) + ";" + os.environ["PATH"]

    try:
        lib = ctypes.CDLL(str(dll_path))
        
        # InitPipeline
        lib.InitPipeline.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
        lib.InitPipeline.restype = ctypes.c_int
        
        # ProcessImage
        lib.ProcessImage.argtypes = [
            ctypes.c_wchar_p, 
            ctypes.POINTER(ctypes.POINTER(FaceInfo)), 
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int # face_type
        ]
        lib.ProcessImage.restype = ctypes.c_int
        
        # FreeFaceResults
        lib.FreeFaceResults.argtypes = [ctypes.POINTER(FaceInfo), ctypes.c_int]
        lib.FreeFaceResults.restype = None

        return lib
    except Exception as e:
        print(f"Failed to load DLL: {e}")
        return None

def cv2_imread(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    except Exception:
        return None

def test():
    lib = load_lib()
    if not lib:
        return

    # Init
    model_dir = Path(__file__).parent.parent / "assets/models"
    if not model_dir.exists():
         print(f"Models directory not found: {model_dir}")
         return
    
    print(f"Loading models from: {model_dir}")
    ret = lib.InitPipeline(str(model_dir), 0) # 0 for GPU 0, -1 for CPU
    if ret != 0:
        print(f"Init failed: {ret}")
        return
    print("Models initialized.")

    # 使用指定的测试图片
    test_image = Path(r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\data\output\cpp版本\S3FD+2DFAN\00100_0.jpg")
    
    if not test_image.exists():
        print(f"测试图片不存在: {test_image}")
    
    img = cv2_imread(str(test_image))
    if img is None:
        print("Failed to read image")
        return

    print(f"Testing image: {test_image}")

    out_faces = ctypes.POINTER(FaceInfo)()
    out_count = ctypes.c_int(0)

    ret = lib.ProcessImage(str(test_image), ctypes.byref(out_faces), ctypes.byref(out_count), 2) # 2 = FULL

    if ret == 0:
        print(f"Detected {out_count.value} faces.")
        for i in range(out_count.value):
            f = out_faces[i]
            print(f"Face {i}: Rect [{f.source_rect[0]:.2f}, {f.source_rect[1]:.2f}, {f.source_rect[2]:.2f}, {f.source_rect[3]:.2f}] Score: {f.detect_score:.4f}")
            
            # Draw landmarks
            for k in range(68):
                x = int(f.landmarks[k*2])
                y = int(f.landmarks[k*2+1])
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                
        lib.FreeFaceResults(out_faces, out_count)
        
        cv2.imwrite("verify_output.jpg", img)
        print("Output saved to verify_output.jpg")

    else:
        print(f"Detection failed: {ret}")

if __name__ == "__main__":
    test()
