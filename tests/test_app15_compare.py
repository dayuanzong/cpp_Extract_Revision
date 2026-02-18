import ctypes
import os
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

REF_DIR = Path(r"d:\Program Files\face_classification\TensorFlow_Extract_Revision\原代码提供参考")
DFLIMG_DIR = REF_DIR / "原始图片类"
FACELIB_DIR = REF_DIR / "facelib"

for p in [REF_DIR, DFLIMG_DIR, FACELIB_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from DFLJPG import DFLJPG
import LandmarksProcessor as LP


def load_dll():
    dll_path = ROOT_DIR / "build" / "Release" / "FaceExtractorDLL.dll"
    if not dll_path.exists():
        dll_path = ROOT_DIR / "build" / "FaceExtractorDLL.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"DLL not found: {dll_path}")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_path.parent))

    return ctypes.cdll.LoadLibrary(str(dll_path))


def read_file_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def main():
    img_path = Path(r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\data\output\原版\00007_1.jpg")
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    jpg_bytes = read_file_bytes(img_path)

    lib = load_dll()
    lib.ExtractApp15Jpeg.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.ExtractApp15Jpeg.restype = ctypes.c_int
    lib.FreeImageBuffer.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    lib.FreeImageBuffer.restype = None

    in_jpg = (ctypes.c_ubyte * len(jpg_bytes)).from_buffer_copy(jpg_bytes)
    out_ptr = ctypes.POINTER(ctypes.c_ubyte)()
    out_size = ctypes.c_int(0)
    ret = lib.ExtractApp15Jpeg(in_jpg, len(jpg_bytes), ctypes.byref(out_ptr), ctypes.byref(out_size))
    if ret != 0 or not out_ptr or out_size.value <= 0:
        raise RuntimeError(f"ExtractApp15Jpeg 失败: {ret}")

    try:
        app15_bytes = ctypes.string_at(out_ptr, out_size.value)
    finally:
        lib.FreeImageBuffer(out_ptr)

    meta_cpp = pickle.loads(app15_bytes)

    dfl = DFLJPG.load(str(img_path))
    if dfl is None:
        raise RuntimeError("DFLJPG.load 失败")
    meta_py = dfl.get_dict()

    landmarks_cpp = np.array(meta_cpp.get("landmarks"))
    landmarks_py = np.array(meta_py.get("landmarks"))
    landmarks_same = np.array_equal(landmarks_cpp, landmarks_py)

    source_cpp = np.array(meta_cpp.get("source_landmarks"))
    source_py = np.array(meta_py.get("source_landmarks"))
    source_same = np.array_equal(source_cpp, source_py)

    face_mat_cpp = meta_cpp.get("image_to_face_mat")
    face_mat_py = meta_py.get("image_to_face_mat")
    mat_same = True
    if face_mat_cpp is not None or face_mat_py is not None:
        mat_same = np.allclose(np.array(face_mat_cpp), np.array(face_mat_py))

    angles_py = None
    if landmarks_py is not None and len(landmarks_py) > 0:
        angles_py = LP.estimate_pitch_yaw_roll(landmarks_py)

    angles_cpp = None
    if landmarks_cpp is not None and len(landmarks_cpp) > 0:
        angles_cpp = LP.estimate_pitch_yaw_roll(landmarks_cpp)

    print("APP15 读取: OK")
    print("landmarks 一致:", "OK" if landmarks_same else "FAIL")
    print("source_landmarks 一致:", "OK" if source_same else "FAIL")
    print("image_to_face_mat 一致:", "OK" if mat_same else "FAIL")
    print("姿态角度 (Python DFLJPG):", angles_py)
    print("姿态角度 (C++ APP15):", angles_cpp)


if __name__ == "__main__":
    main()
