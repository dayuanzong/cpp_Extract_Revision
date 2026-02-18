import sys
import os
import ctypes
import argparse
import csv
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Setup paths
root_dir = Path(__file__).parent.parent
dll_path = root_dir / "bin" / "FaceExtractorDLL.dll"

# Structures
class FaceRectOnly(ctypes.Structure):
    _fields_ = [("x1", ctypes.c_float),
                ("y1", ctypes.c_float),
                ("x2", ctypes.c_float),
                ("y2", ctypes.c_float)]

class FaceStruct(ctypes.Structure):
    _fields_ = [("rect", FaceRectOnly),
                ("score", ctypes.c_float)]

class Point2fStruct(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float)]

# Load DLL
def load_lib():
    if not dll_path.exists():
        print(f"DLL not found at {dll_path}")
        return None
    
    try:
        lib = ctypes.CDLL(str(dll_path))
        
        # InitModels
        lib.InitModels.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
        lib.InitModels.restype = ctypes.c_int
        
        # DetectFaces
        lib.DetectFaces.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), # input_data
            ctypes.c_int, # width
            ctypes.c_int, # height
            ctypes.c_int, # channels
            ctypes.POINTER(FaceStruct), # out_faces
            ctypes.c_int, # max_faces
            ctypes.POINTER(ctypes.c_int) # out_count
        ]
        lib.DetectFaces.restype = ctypes.c_int
        
        # ExtractLandmarks
        lib.ExtractLandmarks.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            FaceRectOnly, # face_rect
            ctypes.POINTER(Point2fStruct) # out_landmarks
        ]
        lib.ExtractLandmarks.restype = ctypes.c_int
        
        return lib
    except Exception as e:
        print(f"Failed to load DLL: {e}")
        return None

def cv2_imread(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    except Exception:
        return None

def compare_two_images(image_a: Path, image_b: Path, output_csv: Path):
    img_a = cv2_imread(str(image_a))
    img_b = cv2_imread(str(image_b))
    if img_a is None or img_b is None:
        raise RuntimeError("无法读取输入图片")

    if img_a.shape != img_b.shape:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_a", "image_b", "status", "shape_a", "shape_b"])
            writer.writerow([str(image_a), str(image_b), "shape_mismatch", str(img_a.shape), str(img_b.shape)])
        return

    diff = img_a.astype(np.float32) - img_b.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "height", "width", "channels", "mae", "mse", "max_abs"])
        writer.writerow([str(image_a), str(image_b), img_a.shape[0], img_a.shape[1], img_a.shape[2], mae, mse, max_abs])


def _to_array(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.array(value)
        except Exception:
            return None
    return None


def _summarize_value(value):
    arr = _to_array(value)
    if arr is not None and arr.dtype != object:
        return {
            "type": "ndarray",
            "dtype": str(arr.dtype),
            "shape": str(arr.shape),
            "min": float(np.min(arr)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
            "mean": float(np.mean(arr)) if arr.size else None,
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(list(value.keys())),
        }
    return {
        "type": type(value).__name__,
        "value": value,
    }


def _compare_values(a, b):
    arr_a = _to_array(a)
    arr_b = _to_array(b)
    if arr_a is not None and arr_b is not None and arr_a.dtype != object and arr_b.dtype != object:
        if arr_a.shape != arr_b.shape:
            return {
                "status": "shape_mismatch",
                "shape_a": str(arr_a.shape),
                "shape_b": str(arr_b.shape),
                "detail": "",
            }
        diff = arr_a.astype(np.float64) - arr_b.astype(np.float64)
        return {
            "status": "diff",
            "shape_a": str(arr_a.shape),
            "shape_b": str(arr_b.shape),
            "detail": json.dumps({
                "max_abs": float(np.max(np.abs(diff))) if diff.size else None,
                "mean_abs": float(np.mean(np.abs(diff))) if diff.size else None,
                "mean": float(np.mean(diff)) if diff.size else None,
            }, ensure_ascii=False),
        }
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        only_a = sorted(list(keys_a - keys_b))
        only_b = sorted(list(keys_b - keys_a))
        return {
            "status": "dict_diff" if only_a or only_b else "equal",
            "shape_a": "",
            "shape_b": "",
            "detail": json.dumps({"only_in_a": only_a, "only_in_b": only_b}, ensure_ascii=False),
        }
    return {
        "status": "equal" if a == b else "diff",
        "shape_a": "",
        "shape_b": "",
        "detail": json.dumps({"a": a, "b": b}, ensure_ascii=False),
    }


def compare_metadata(image_a: Path, image_b: Path, output_csv: Path):
    root_dir = Path(__file__).parent.parent
    libs_dir = root_dir / "sdk" / "_libs"
    sys.path.append(str(libs_dir))
    from DFLIMG import DFLJPG

    dfl_a = DFLJPG.load(str(image_a))
    dfl_b = DFLJPG.load(str(image_b))
    if dfl_a is None or dfl_b is None:
        raise RuntimeError("无法读取 DFLJPG 元数据")

    dict_a = dfl_a.get_dict() or {}
    dict_b = dfl_b.get_dict() or {}

    keys = sorted(list(set(dict_a.keys()) | set(dict_b.keys())))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "key",
            "status",
            "type_a",
            "type_b",
            "shape_a",
            "shape_b",
            "summary_a",
            "summary_b",
            "detail_diff",
        ])
        for key in keys:
            a = dict_a.get(key, None)
            b = dict_b.get(key, None)
            if key not in dict_a:
                writer.writerow([key, "only_in_b", "", type(b).__name__, "", "", "", json.dumps(_summarize_value(b), ensure_ascii=False), ""])
                continue
            if key not in dict_b:
                writer.writerow([key, "only_in_a", type(a).__name__, "", "", "", json.dumps(_summarize_value(a), ensure_ascii=False), "", ""])
                continue
            summary_a = _summarize_value(a)
            summary_b = _summarize_value(b)
            cmp_result = _compare_values(a, b)
            writer.writerow([
                key,
                cmp_result["status"],
                summary_a.get("type", ""),
                summary_b.get("type", ""),
                cmp_result.get("shape_a", ""),
                cmp_result.get("shape_b", ""),
                json.dumps(summary_a, ensure_ascii=False),
                json.dumps(summary_b, ensure_ascii=False),
                cmp_result.get("detail", ""),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_images", action="store_true")
    parser.add_argument("--compare_meta", action="store_true")
    parser.add_argument("--image_a", type=str, default="")
    parser.add_argument("--image_b", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="")
    args = parser.parse_args()

    if args.compare_images:
        if not args.image_a or not args.image_b:
            raise SystemExit("需要提供 --image_a 和 --image_b")
        output_csv = Path(args.output_csv) if args.output_csv else Path(__file__).parent / "image_compare.csv"
        compare_two_images(Path(args.image_a), Path(args.image_b), output_csv)
        print(f"已写入 CSV: {output_csv}")
        return
    if args.compare_meta:
        if not args.image_a or not args.image_b:
            raise SystemExit("需要提供 --image_a 和 --image_b")
        output_csv = Path(args.output_csv) if args.output_csv else Path(__file__).parent / "logs" / "metadata_compare.csv"
        compare_metadata(Path(args.image_a), Path(args.image_b), output_csv)
        print(f"已写入 CSV: {output_csv}")
        return

    root_dir = Path(__file__).parent.parent
    
    # 1. Verify File Consistency
    # Python Output: data/output/python版本/aligned/低方差
    # C++ Output: data/output/cpp版本/aligned/低方差
    
    py_output_dir = root_dir / "data/output/python版本/aligned/低方差"
    cpp_output_dir = root_dir / "data/output/cpp版本/aligned/低方差"
    
    if not py_output_dir.exists():
        print(f"Python output dir not found: {py_output_dir}")
        return

    if not cpp_output_dir.exists():
        print(f"C++ output dir not found: {cpp_output_dir}")
        return

    print(f"Comparing:\n  Python: {py_output_dir}\n  C++:    {cpp_output_dir}")
    
    py_files = {f.name for f in py_output_dir.glob("*.jpg")}
    cpp_files = {f.name for f in cpp_output_dir.glob("*.jpg")}
    
    print(f"\nPython count: {len(py_files)}")
    print(f"C++ count:    {len(cpp_files)}")
    
    common = py_files.intersection(cpp_files)
    only_py = py_files - cpp_files
    only_cpp = cpp_files - py_files
    
    print(f"Common: {len(common)}")
    if only_py:
        print(f"Only in Python: {len(only_py)}")
        print("  " + ", ".join(list(only_py)[:5]))
    if only_cpp:
        print(f"Only in C++: {len(only_cpp)}")
        print("  " + ", ".join(list(only_cpp)[:5]))

    # 2. Verify Landmarks on Common Files
    print("\nVerifying Landmarks on Common Files...")
    
    # Setup DFLIMG
    libs_dir = root_dir / "sdk" / "_libs"
    sys.path.append(str(libs_dir))
    try:
        from DFLIMG import DFLJPG
    except ImportError:
        sys.path.append(str(libs_dir / "facelib"))
        from DFLIMG import DFLJPG
        
    # Init C++ Lib
    lib = load_lib()
    if not lib:
        return
        
    model_dir = root_dir / "assets" / "models"
    lib.InitModels(str(model_dir), 0)
    
    # Process sample of common files (or all)
    # Since we need to run detection on ORIGINAL images to get C++ landmarks,
    # we need to map aligned filename back to original filename.
    # usually 00001_0.jpg -> 00001.png (in data/input/包含面部图片)
    
    input_dir = root_dir / "data/input/包含面部图片"
    
    mse_list = []
    
    # Limit to 50 for speed if many
    sample_files = sorted(list(common))
    if len(sample_files) > 50:
        print("Sampling 50 files for verification...")
        sample_files = sample_files[:50]
        
    for fname in tqdm(sample_files):
        # 1. Load GT Landmarks from Python result
        py_path = py_output_dir / fname
        dflimg = DFLJPG.load(str(py_path))
        if not dflimg or not dflimg.has_data():
            continue
            
        gt_lm = dflimg.get_source_landmarks()
        gt_rect = dflimg.get_source_rect() # (l,t,r,b)
        if gt_lm is None:
            continue
            
        # 2. Find Original Image
        # fname is like 00098_0.jpg
        stem = fname.split('_')[0]
        # extensions?
        orig_path = None
        for ext in ['.png', '.jpg', '.bmp']:
            p = input_dir / (stem + ext)
            if p.exists():
                orig_path = p
                break
        
        if not orig_path:
            continue
            
        # 3. Run C++ Inference
        img = cv2_imread(str(orig_path))
        if img is None:
            continue
        h, w, c = img.shape
        
        max_faces = 10
        faces = (FaceStruct * max_faces)()
        count = ctypes.c_int()
        
        lib.DetectFaces(img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), w, h, c, faces, max_faces, ctypes.byref(count))
        
        # Find the matching face
        best_lm = None
        min_dist = float('inf')
        
        for i in range(count.value):
            f = faces[i]
            # Check overlap with GT rect
            l, t, r, b = f.rect.x1, f.rect.y1, f.rect.x2, f.rect.y2
            gl, gt, gr, gb = gt_rect
            
            # IoU or simple overlap
            dx = min(r, gr) - max(l, gl)
            dy = min(b, gb) - max(t, gt)
            
            if dx > 0 and dy > 0:
                # Matched
                landmarks = (Point2fStruct * 68)()
                ret_lm = lib.ExtractLandmarks(img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), w, h, c, f.rect, landmarks)
                if ret_lm == 0:
                    lm_np = np.array([(p.x, p.y) for p in landmarks], dtype=np.float32)
                    
                    # Calculate MSE with GT
                    mse = np.mean((lm_np - gt_lm)**2)
                    if mse < min_dist:
                        min_dist = mse
                        best_lm = lm_np
        
        if best_lm is not None:
            mse_list.append(min_dist)
            
    if mse_list:
        avg_mse = np.mean(mse_list)
        print(f"\nAverage Landmarks MSE: {avg_mse:.4f}")
        print(f"Max MSE: {np.max(mse_list):.4f}")
        print(f"Min MSE: {np.min(mse_list):.4f}")
        if avg_mse < 5.0: # Arbitrary threshold, typically < 1-5 is good for pixels
            print("verification Passed: Landmarks match closely.")
        else:
            print("Verification Warning: High MSE.")
    else:
        print("No valid comparisons made.")

if __name__ == "__main__":
    main()
