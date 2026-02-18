import ctypes
import os
import pickle
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
LIBS_DIR = ROOT_DIR / "_libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from DFLIMG.DFLJPG import DFLJPG


def load_dll():
    root_dir = ROOT_DIR
    dll_path = root_dir / "build" / "Release" / "FaceExtractorDLL.dll"
    if not dll_path.exists():
        dll_path = root_dir / "build" / "FaceExtractorDLL.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"DLL not found: {dll_path}")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_path.parent))

    return ctypes.cdll.LoadLibrary(str(dll_path))


def make_test_image():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (246, 246), (0, 255, 0), 2)
    cv2.circle(img, (128, 128), 60, (255, 0, 0), 3)
    return img


def main():
    output_dir = ROOT_DIR / "output_test"
    output_dir.mkdir(exist_ok=True)

    lib = load_dll()
    lib.InsertApp15Jpeg.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.InsertApp15Jpeg.restype = ctypes.c_int
    lib.FreeImageBuffer.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    lib.FreeImageBuffer.restype = None

    img = make_test_image()
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    jpg_bytes = buf.tobytes()

    meta = {
        "face_type": "full_face",
        "landmarks": [[10.0, 20.0], [30.0, 40.0]],
        "source_rect": [1, 2, 3, 4],
    }
    app15_bytes = pickle.dumps(meta)

    in_jpg = (ctypes.c_ubyte * len(jpg_bytes)).from_buffer_copy(jpg_bytes)
    in_app15 = (ctypes.c_ubyte * len(app15_bytes)).from_buffer_copy(app15_bytes)
    out_ptr = ctypes.POINTER(ctypes.c_ubyte)()
    out_size = ctypes.c_int(0)

    ret = lib.InsertApp15Jpeg(in_jpg, len(jpg_bytes), in_app15, len(app15_bytes), ctypes.byref(out_ptr), ctypes.byref(out_size))
    if ret != 0 or not out_ptr or out_size.value <= 0:
        raise RuntimeError(f"InsertApp15Jpeg 失败: {ret}")

    try:
        injected_bytes = ctypes.string_at(out_ptr, out_size.value)
    finally:
        lib.FreeImageBuffer(out_ptr)

    dfl = DFLJPG.load_from_memory(injected_bytes, "cpp_app15.jpg")
    if dfl is None:
        raise RuntimeError("DFLJPG 无法解析 C++ 生成的 APP15 JPG")

    loaded_meta = dfl.get_dict()
    meta_ok = loaded_meta == meta

    dumped_bytes = dfl.dump()
    bytes_same = dumped_bytes == injected_bytes

    out_path = output_dir / "cpp_app15_saved.jpg"
    with open(out_path, "wb") as f:
        f.write(dumped_bytes)

    reload_dfl = DFLJPG.load(str(out_path))
    reload_ok = reload_dfl is not None and reload_dfl.get_dict() == meta

    if not bytes_same:
        img_a = cv2.imdecode(np.frombuffer(injected_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(np.frombuffer(dumped_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        pixel_ok = img_a is not None and img_b is not None and img_a.shape == img_b.shape and np.array_equal(img_a, img_b)
    else:
        pixel_ok = True

    print("DFLJPG 识别:", "OK" if meta_ok else "FAIL")
    print("保存一致性:", "OK" if reload_ok else "FAIL")
    print("字节一致性:", "OK" if bytes_same else "FAIL")
    print("像素一致性:", "OK" if pixel_ok else "FAIL")
    print(f"输出路径: {out_path}")


if __name__ == "__main__":
    main()
