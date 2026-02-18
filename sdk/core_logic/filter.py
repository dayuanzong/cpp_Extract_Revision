
import sys
import os
import argparse
from pathlib import Path
import shutil
import re
import cv2
import numpy as np
import traceback

from . import utils

# Import libs
try:
    from DFLIMG import DFLJPG
    from facelib import LandmarksProcessor
except ImportError as e:
    print(f"Failed to import libraries in filter: {e}")

def read_image(path):
    try:
        with open(str(path), "rb") as f:
            bytes_data = bytearray(f.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

def parse_select_tags(select_poses):
    if not select_poses:
        return []
    tags = [t for t in re.split(r"[,\s，]+", select_poses) if t]
    allowed = {"抬头", "低头", "向左", "向右"}
    return [t for t in tags if t in allowed]

def classify_pose(pitch_deg, yaw_deg, selected_tags, pitch_threshold, yaw_threshold):
    if not selected_tags:
        return None
    priority = ["抬头", "低头", "向左", "向右"]
    for tag in priority:
        if tag not in selected_tags:
            continue
        if tag == "抬头" and pitch_deg > pitch_threshold:
            return tag
        if tag == "低头" and pitch_deg < -pitch_threshold:
            return tag
        if tag == "向左" and yaw_deg < -yaw_threshold:
            return tag
        if tag == "向右" and yaw_deg > yaw_threshold:
            return tag
    return None

def is_mouth_open(landmarks, threshold):
    if landmarks is None or len(landmarks) < 67:
        return False
    mouth_open_diff = landmarks[66][1] - landmarks[62][1]
    return mouth_open_diff > threshold

def compute_blur_variance(image, landmarks):
    if image is None or landmarks is None:
        return None
    try:
        mask = LandmarksProcessor.get_image_hull_mask(image.shape, landmarks)
        if mask is None:
            return None
        mask_bool = mask[..., 0] > 0.5
        if not np.any(mask_bool):
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap[mask_bool].var())
    except Exception:
        return None

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def move_to_dir(src, dst_dir):
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst
    stem = src.stem
    suffix = src.suffix
    index = 1
    while True:
        candidate = dst_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
        index += 1

def to_5pts_from_68(lmk68):
    if lmk68 is None or len(lmk68) != 68:
        return None
    le = lmk68[36:42].mean(axis=0)
    re = lmk68[42:48].mean(axis=0)
    nose = lmk68[30]
    lm = lmk68[48]
    rm = lmk68[54]
    return np.stack([le, re, nose, lm, rm]).astype(np.float32)

def normalize_vec(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n != 0 else None

def embedding_from_landmarks(img, lmk68, rec, face_align):
    kps5 = to_5pts_from_68(lmk68)
    if kps5 is None:
        return None
    aligned = face_align.norm_crop(img, landmark=kps5, image_size=112)
    feat = rec.get_feat(aligned)
    if isinstance(feat, list):
        feat = feat[0]
    v = np.array(feat, dtype=np.float32).flatten()
    return normalize_vec(v)

def collect_embeddings(search_dir, rec, face_align):
    sum_vec = None
    count = 0
    exts = {".jpg", ".jpeg"}
    for p in sorted(search_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        dflimg = DFLJPG.load(str(p))
        if dflimg is None or not dflimg.has_data():
            continue
        landmarks = dflimg.get_landmarks()
        if landmarks is None:
            continue
        img = read_image(p)
        if img is None:
            continue
        emb = embedding_from_landmarks(img, landmarks, rec, face_align)
        if emb is None:
            continue
        if sum_vec is None:
            sum_vec = emb.copy()
        else:
            sum_vec += emb
        count += 1
    return sum_vec, count

def run_target_filter(args, input_dir, output_dir):
    if not args.reference_dir:
        print("指定人物筛选需要参考目录")
        return

    ref_dir = Path(args.reference_dir)
    if not ref_dir.exists():
        print(f"参考目录不存在: {ref_dir}")
        return

    try:
        import onnxruntime as ort
        from insightface.model_zoo import get_model
        from insightface.utils import face_align
    except Exception as e:
        print(f"初始化指定人物筛选失败: {e}")
        return

    model_path = Path(__file__).parent.parent / "models" / "w600k_r50.onnx"
    if not model_path.exists():
        model_path = Path("models/w600k_r50.onnx")
    if not model_path.exists():
        print(f"识别模型不存在: {model_path}")
        return

    providers = ["CPUExecutionProvider"]
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider"]

    rec = get_model(str(model_path), providers=providers)
    rec.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)

    reference_embs = []
    reference_names = []
    multi_target_active = False
    subdirs = [d for d in ref_dir.iterdir() if d.is_dir()]
    if args.multi_target and subdirs:
        for sd in subdirs:
            sum_vec, count = collect_embeddings(sd, rec, face_align)
            if count > 0 and sum_vec is not None:
                mean_emb = sum_vec / max(1, count)
                emb = normalize_vec(mean_emb)
                if emb is not None:
                    reference_embs.append(emb)
                    reference_names.append(sd.name)
        if reference_embs:
            multi_target_active = True

    reference_emb = None
    if not multi_target_active:
        sum_vec, count = collect_embeddings(ref_dir, rec, face_align)
        if count > 0 and sum_vec is not None:
            mean_emb = sum_vec / max(1, count)
            reference_emb = normalize_vec(mean_emb)

    if multi_target_active:
        print(f"参考人物数量: {len(reference_embs)}，阈值: {args.sim_threshold}")
    elif reference_emb is not None:
        print(f"参考人脸数量: 1，阈值: {args.sim_threshold}")
    else:
        print("未能从参考目录提取到有效人脸特征")
        return

    image_files = [p for p in input_dir.glob("*.jpg")] + [p for p in input_dir.glob("*.jpeg")]
    if not image_files:
        print("未找到JPG文件")
        return

    pick_target = getattr(args, "target_pick", "target") != "non_target"
    moved_count = 0

    for path in image_files:
        dflimg = DFLJPG.load(str(path))
        if dflimg is None or not dflimg.has_data():
            continue
        landmarks = dflimg.get_landmarks()
        if landmarks is None:
            continue
        img = read_image(path)
        if img is None:
            continue
        emb = embedding_from_landmarks(img, landmarks, rec, face_align)
        if emb is None:
            continue

        if multi_target_active:
            best_idx = None
            best_sim = -1.0
            for i, ref_emb in enumerate(reference_embs):
                sim = float(np.dot(emb, ref_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx is not None and best_sim >= args.sim_threshold:
                if pick_target:
                    move_to_dir(path, output_dir / reference_names[best_idx])
                    moved_count += 1
            else:
                if not pick_target:
                    move_to_dir(path, output_dir / "非目标人物")
                    moved_count += 1
        else:
            sim = float(np.dot(emb, reference_emb))
            if sim >= args.sim_threshold:
                if pick_target:
                    move_to_dir(path, output_dir / "目标人物")
                    moved_count += 1
            else:
                if not pick_target:
                    move_to_dir(path, output_dir / "非目标人物")
                    moved_count += 1

    label = "目标" if pick_target else "非目标"
    print(f"{label}:{moved_count} 张")

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def run(args_dict, log_queue=None):
    if log_queue:
        utils.setup_process_logging(log_queue)
        
    args = Args(**args_dict)
    
    input_dir = Path(args.input_dir.strip())
    output_dir = Path(args.output_dir.strip()) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"目录不存在: {input_dir}")
        return

    # Logic from original script to set defaults based on mode
    if args.mode:
        if args.mode == "pose":
            if not args.select_poses: args.select_poses = "抬头,低头,向左,向右"
            args.select_mouth_open = False
            args.blur_analysis = False
        elif args.mode == "mouth":
            args.select_poses = ""
            args.select_mouth_open = True
            args.blur_analysis = False
        elif args.mode == "blur":
            args.select_poses = ""
            args.select_mouth_open = False
            args.blur_analysis = True
        elif args.mode == "target":
            run_target_filter(args, input_dir, output_dir)
            return

    enabled_count = 0
    enabled_count += 1 if args.select_poses else 0
    enabled_count += 1 if args.select_mouth_open else 0
    enabled_count += 1 if args.blur_analysis else 0
    if enabled_count == 0:
        print("未启用任何筛选条件")
        return
    if enabled_count > 1:
        print("角度、张嘴、模糊不可同时运行")
        return

    pose_tags = parse_select_tags(args.select_poses)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if not image_files:
        print("未找到JPG文件")
        return

    stats = {
        "pose": 0,
        "mouth": 0,
        "blur_low": 0,
        "blur_mid": 0,
        "blur_high": 0,
        "blur_error": 0,
    }

    print(f"开始处理 {len(image_files)} 张图片...")

    for path in image_files:
        dflimg = DFLJPG.load(str(path))
        if dflimg is None or not dflimg.has_data():
            continue
        landmarks = dflimg.get_landmarks()
        if landmarks is None:
            continue
        
        if args.debug:
            debug_image = read_image(path)
            if debug_image is not None:
                LandmarksProcessor.draw_landmarks(debug_image, landmarks, transparent_mask=True)
                debug_dir = output_dir / "_debug"
                ensure_dir(debug_dir)
                debug_file = debug_dir / path.name
                cv2.imencode(".jpg", debug_image)[1].tofile(str(debug_file))

        if pose_tags:
            shape = dflimg.get_shape()
            img_size = shape[1] if shape and len(shape) > 1 else 0
            try:
                pitch, yaw, _ = LandmarksProcessor.estimate_pitch_yaw_roll(landmarks, img_size)
                pitch_deg = float(pitch * 57.29578)
                yaw_deg = float(yaw * 57.29578)
                tag = classify_pose(pitch_deg, yaw_deg, pose_tags, args.pose_pitch_threshold, args.pose_yaw_threshold)
                if tag:
                    move_to_dir(path, output_dir / tag)
                    stats["pose"] += 1
            except Exception:
                pass

        if args.select_mouth_open:
            if is_mouth_open(landmarks, args.mouth_open_threshold):
                move_to_dir(path, output_dir / "张嘴")
                stats["mouth"] += 1

        if args.blur_analysis:
            image = read_image(path)
            variance = compute_blur_variance(image, landmarks)
            if variance is None:
                move_to_dir(path, output_dir / "模糊_错误")
                stats["blur_error"] += 1
            elif variance < args.blur_low_threshold:
                move_to_dir(path, output_dir / "模糊_低")
                stats["blur_low"] += 1
            elif variance < args.blur_high_threshold:
                move_to_dir(path, output_dir / "模糊_中")
                stats["blur_mid"] += 1
            else:
                move_to_dir(path, output_dir / "模糊_高")
                stats["blur_high"] += 1

    print(
        f"角度:{stats['pose']} 张, 张嘴:{stats['mouth']} 张, "
        f"模糊_低:{stats['blur_low']} 张, 模糊_中:{stats['blur_mid']} 张, "
        f"模糊_高:{stats['blur_high']} 张, 模糊_错误:{stats['blur_error']} 张"
    )
