import ctypes
from ctypes import c_int, c_float, c_void_p, c_wchar_p, POINTER, Structure
import os
from pathlib import Path

class FaceInfo(Structure):
    _fields_ = [
        ("jpg_data", ctypes.POINTER(ctypes.c_ubyte)),
        ("jpg_size", c_int),
        ("landmarks", c_float * 136),
        ("aligned_landmarks", c_float * 136),
        ("embedding_dim", c_int),
        ("embedding", c_float * 512),
        ("target_index", c_int),
        ("target_sim", c_float),
        ("is_target", c_int), # bool is int in C interface usually
        ("source_rect", c_float * 4),
        ("detect_score", c_float),
        ("blur_variance", c_float),
        ("pose_tag", ctypes.c_char * 32),
        ("pitch", c_float),
        ("yaw", c_float),
        ("roll", c_float),
        ("blur_class", c_int),
        ("mouth_value", c_float),
        ("mouth_open", c_int),
        ("valid", c_int)
    ]

def main():
    dll_path = Path(r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\cpp_Extract_Revision\bin\FaceExtractorDLL.dll")
    if not dll_path.exists():
        print(f"DLL not found at {dll_path}")
        return

    try:
        lib = ctypes.CDLL(str(dll_path))
    except Exception as e:
        print(f"Failed to load DLL: {e}")
        return

    # Define signatures
    lib.InitPipelineEx.argtypes = [c_wchar_p, c_int, c_wchar_p, c_wchar_p, c_wchar_p]
    lib.InitPipelineEx.restype = c_int

    lib.ProcessImage.argtypes = [c_wchar_p, POINTER(POINTER(FaceInfo)), POINTER(c_int), c_int]
    lib.ProcessImage.restype = c_int

    lib.FreeFaceResults.argtypes = [POINTER(FaceInfo), c_int]
    lib.FreeFaceResults.restype = None

    # Paths
    model_dir = r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\cpp_Extract_Revision\assets\models"
    # det_10g.onnx for detection
    s3fd_path = str(Path(model_dir) / "det_10g.onnx")
    # 2d106det.onnx for landmarks
    fan_path = str(Path(model_dir) / "2d106det.onnx")
    # w600k_r50.onnx for recognition (optional)
    insight_path = str(Path(model_dir) / "w600k_r50.onnx")

    print("Initializing pipeline...")
    print(f"S3FD (SCRFD): {s3fd_path}")
    print(f"FAN (Insight): {fan_path}")
    
    ret = lib.InitPipelineEx(model_dir, 0, s3fd_path, fan_path, insight_path)
    if ret != 0:
        print(f"InitPipelineEx failed with code {ret}")
        return
    print("Pipeline initialized.")

    # Test image
    img_path = r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\TensorFlow_Extract\data\包含面部图片\00001.png"
    if not os.path.exists(img_path):
        print(f"Test image not found at {img_path}")
        return

    print(f"Processing image {img_path}...")
    faces = POINTER(FaceInfo)()
    count = c_int(0)
    
    ret = lib.ProcessImage(img_path, ctypes.byref(faces), ctypes.byref(count), 2) # 2 = FULL
    if ret != 0:
        print(f"ProcessImage failed with code {ret}")
        return

    print(f"Found {count.value} faces.")
    
    for i in range(count.value):
        face = faces[i]
        print(f"Face {i}: Score={face.detect_score:.4f}, Blur={face.blur_variance:.4f}")
        print(f"  Rect: {list(face.source_rect)}")
        # Check landmarks (Nose tip: 30, Left Eye Corner: 36)
        # Index 30 -> 60, 61
        # Index 36 -> 72, 73
        lms = list(face.landmarks)
        print(f"  Nose (30): [{lms[60]:.2f}, {lms[61]:.2f}]")
        print(f"  L-Eye (36): [{lms[72]:.2f}, {lms[73]:.2f}]")

    lib.FreeFaceResults(faces, count)
    print("Done.")

if __name__ == "__main__":
    main()
