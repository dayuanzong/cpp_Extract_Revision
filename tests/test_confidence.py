import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Add sdk/_libs to sys.path
# tests/test_confidence.py -> tests -> root -> sdk -> _libs
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
libs_dir = root_dir / "sdk" / "_libs"
sys.path.insert(0, str(libs_dir))

try:
    from FaceExtractorWrapper import FaceExtractorWrapper
    from DFLIMG import DFLJPG
except ImportError:
    print("Warning: Failed to import dependencies from sdk/_libs")
    FaceExtractorWrapper = None
    DFLJPG = None

def read_image(path):
    try:
        with open(str(path), "rb") as f:
            bytes_data = bytearray(f.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"读取失败: {path} {e}")
        return None

def normalize_vec(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n != 0 else None

def collect_images(p):
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [x for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in exts]

def best_face_landmarks(wrapper, img_path, face_type_int):
    faces = wrapper.process_image(img_path, face_type_int)
    if not faces:
        return None
    best_face = None
    best_area = 0.0
    for face in faces:
        rect = face["rect"]
        area = max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])
        if area > best_area:
            best_area = area
            best_face = face
    if best_face is None:
        return None
    return best_face.get("landmarks")

def embedding_from_image(wrapper, img_path, img, landmarks):
    if img is None or landmarks is None or len(landmarks) != 68:
        return None
    emb = wrapper.extract_embedding(img, landmarks)
    if emb is None:
        return None
    return normalize_vec(np.array(emb, dtype=np.float32))

def embedding_from_path(wrapper, img_path, face_type_int):
    if DFLJPG is None:
        return None
    dflimg = DFLJPG.load(str(img_path))
    if dflimg is None or not dflimg.has_data():
        return None
    landmarks = dflimg.get_landmarks()
    if landmarks is None or len(landmarks) != 68:
        return None
    img = read_image(img_path)
    if img is None:
        return None
    return embedding_from_image(wrapper, img_path, img, landmarks)

def build_reference_embedding(wrapper, ref_path, face_type_int):
    sum_vec = None
    count = 0
    for p in collect_images(ref_path):
        emb = embedding_from_path(wrapper, p, face_type_int)
        if emb is None:
            continue
        if sum_vec is None:
            sum_vec = emb.copy()
        else:
            sum_vec += emb
        count += 1
    if count == 0 or sum_vec is None:
        return None, 0
    mean_emb = sum_vec / max(1, count)
    return normalize_vec(mean_emb), count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default=str(root_dir / "assets" / "models"))
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--face_type", type=int, default=2)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"模型目录不存在: {models_dir}")
        return

    wrapper = FaceExtractorWrapper(models_dir)

    ref_path = Path(args.reference_dir)
    if not ref_path.exists():
        print(f"参考目录不存在: {ref_path}")
        return

    ref_emb, ref_count = build_reference_embedding(wrapper, ref_path, args.face_type)
    if ref_emb is None:
        print("未能从参考目录得到有效 embedding")
        return

    print(f"参考图片数量: {ref_count}")

    target_paths = []
    for t in args.targets:
        for p in t.split(';'):
            p = p.strip().strip('"')
            if not p:
                continue
            tp = Path(p)
            if tp.exists():
                target_paths.extend(collect_images(tp))
            else:
                print(f"目标不存在: {tp}")

    for p in target_paths:
        # Check pose if possible
        faces = wrapper.process_image(p, args.face_type)
        pose_info = "Unknown"
        if faces:
            # Find best face
            best_face = None
            best_area = 0.0
            for face in faces:
                rect = face["rect"]
                area = max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])
                if area > best_area:
                    best_area = area
                    best_face = face
            if best_face:
                pose_info = best_face.get("pose", "Unknown")

        emb = embedding_from_path(wrapper, p, args.face_type)
        if emb is None:
            print(f"{p} 置信度: 无法提取, 姿态: {pose_info}")
            continue
        sim = float(np.dot(emb, ref_emb))
        print(f"{p} 置信度: {sim:.6f}, 姿态: {pose_info}")

if __name__ == "__main__":
    main()
