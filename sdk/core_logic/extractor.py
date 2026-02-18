
import sys
import os
import shutil
import math
from pathlib import Path
from tqdm import tqdm
import threading
import queue
import concurrent.futures
import time
import traceback
import multiprocessing
import numpy as np
import cv2

# Disable albumentations version check
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
os.environ.setdefault("ORT_LOGGING_SEVERITY_LEVEL", "4")

from . import utils

# Import libs
try:
    from core import np_compat
    from core import mathlib
    from DFLIMG import DFLJPG
    from facelib import FaceType
    from facelib import LandmarksProcessor
    from FaceExtractorWrapper import FaceExtractorWrapper
except ImportError as e:
    print(f"Failed to import libraries in extractor: {e}")

# Thread-safe counters
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

    def add(self, n):
        with self.lock:
            self.value += n
            return self.value
            
    def get(self):
        with self.lock:
            return self.value

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

def compute_blur_variance(image, landmarks):
    if image is None or landmarks is None:
        return None
    try:
        mask = LandmarksProcessor.get_image_hull_mask(image.shape, np.array(landmarks, dtype=np.float32))
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

def is_mouth_open(landmarks, threshold):
    if landmarks is None or len(landmarks) < 67:
        return False
    return (landmarks[66][1] - landmarks[62][1]) > threshold

def classify_pose_tag(pitch_deg, yaw_deg, pitch_threshold, yaw_threshold):
    if pitch_deg > pitch_threshold:
        return "抬头"
    if pitch_deg < -pitch_threshold:
        return "低头"
    if yaw_deg < -yaw_threshold:
        return "向右"
    if yaw_deg > yaw_threshold:
        return "向左"
    return None

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def run(args_dict, log_queue=None):
    if log_queue:
        utils.setup_process_logging(log_queue)
        
    args = Args(**args_dict)
    
    # Global counters for this run
    processed_count = Counter()
    faces_count = Counter()
    
    input_path = Path(args.input_dir)
    output_base = Path(args.output_dir)
    if output_base.name.lower() == "aligned":
        print(f"输出目录不能为 aligned，将改为其上级目录: {output_base.parent}")
        output_base = output_base.parent
    output_path = output_base / "aligned"
    debug_base = input_path.parent
    debug_path = debug_base / "cpp_debug" if args.debug else None
    
    current_dir = Path(__file__).parent.resolve()
    root_dir = current_dir.parent.parent
    
    # Try to find models_dir
    possible_models_dirs = [
        root_dir / "assets" / "models",
        root_dir / "assets" / "models_exported",
        root_dir / "models_exported",
        root_dir / "TF_Extract.dist" / "models",
        root_dir / "main.dist" / "models",
        root_dir / "_libs" / "models"
    ]
    models_dir = root_dir / "assets" / "models"
    for d in possible_models_dirs:
        if d.exists():
            models_dir = d
            break
    
    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        return

    def ensure_empty_dir(path):
        if path.exists():
            has_files = any(p.is_file() for p in path.rglob("*"))
            if has_files:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except Exception as e:
                    print(f"无法清空目录 {path}: {e}")
        path.mkdir(parents=True, exist_ok=True)

    def ensure_dir(path):
        path.mkdir(parents=True, exist_ok=True)

    is_batch = int(getattr(args, "batch_index", 0) or 0) > 0 and int(getattr(args, "batch_total", 0) or 0) > 1

    if args.debug:
        if is_batch:
            ensure_dir(debug_path)
        else:
            ensure_empty_dir(debug_path)
    
    start_time = time.perf_counter()
    
    # Initialize Wrapper
    try:
        print(f"Loading models from: {models_dir}")
        s3fd_path = getattr(args, "s3fd_model_path", "") or ""
        fan_path = getattr(args, "fan_model_path", "") or ""
        rec_path = getattr(args, "rec_model_path", "") or ""
        if s3fd_path:
            print(f"Loading S3FD from override: {s3fd_path}")
        if fan_path:
            print(f"Loading FAN from override: {fan_path}")
        if rec_path:
            print(f"Loading InsightFace from override: {rec_path}")
        wrapper = FaceExtractorWrapper(models_dir, args.gpu_idx, s3fd_path, fan_path, rec_path)
    except Exception as e:
        print(f"初始化模型失败: {e}")
        traceback.print_exc()
        return

    # Filtering thresholds
    blur_thresh = args.blur_thresholds if args.blur_classify else None
    pose_thresh = args.pose_thresholds if args.classify_pose else None
    blur_low = blur_thresh[0] if blur_thresh and len(blur_thresh) > 0 else 10.0
    blur_high = blur_thresh[1] if blur_thresh and len(blur_thresh) > 1 else 20.0
    pitch_th = float(getattr(args, "pose_pitch_threshold", 15.0) or 15.0)
    yaw_th = float(getattr(args, "pose_yaw_threshold", 15.0) or 15.0)
    mouth_th = float(getattr(args, "mouth_open_threshold", 15.0) or 15.0)
    if not hasattr(wrapper, "set_filter_params"):
        print("当前 DLL 缺少 SetFilterParams，前端仅支持 C++ 过滤，已中止")
        return
    align_size = int(getattr(args, "image_size", 256) or 256)
    if align_size <= 0:
        align_size = 256
    if hasattr(wrapper, "set_align_size"):
        wrapper.set_align_size(align_size)
    if hasattr(wrapper, "set_jpeg_quality"):
        wrapper.set_jpeg_quality(int(getattr(args, "jpeg_quality", 90) or 90))
    max_faces = int(getattr(args, "max_faces_from_image", 0) or 0)
    if max_faces < 0:
        max_faces = 0
    if hasattr(wrapper, "set_max_faces"):
        wrapper.set_max_faces(max_faces)
    wrapper.set_filter_params(
        enable_blur=1 if args.blur_classify else 0,
        blur_low=blur_low,
        blur_high=blur_high,
        enable_pose=1 if args.classify_pose else 0,
        pitch_threshold=pitch_th,
        yaw_threshold=yaw_th,
        enable_mouth=1 if args.select_mouth_open else 0,
        mouth_threshold=mouth_th
    )
    
    # Face Type
    face_type_str = getattr(args, "face_type", "full_face")
    try:
        face_type_int = FaceType.fromString(face_type_str)
    except:
        face_type_int = FaceType.FULL
        face_type_str = FaceType.toString(FaceType.FULL)

    reference_emb = None
    reference_embs = []
    reference_names = []
    reference_dirs = []
    target_output_dirs = []
    non_target_dir = None
    multi_target_active = False
    rec = None
    face_align = None

    target_backend_ready = False
    if args.reference_dir:
        if not hasattr(wrapper, "get_embedding_dim") or not hasattr(wrapper, "extract_embedding"):
            print("初始化指定人物提取失败: DLL 缺少 Embedding 接口，已跳过指定人物逻辑")
        else:
            emb_dim = wrapper.get_embedding_dim()
            if emb_dim <= 0:
                print("初始化指定人物提取失败: C++ 识别模型不可用，已跳过指定人物逻辑")
            else:
                def normalize_vec(v):
                    n = np.linalg.norm(v)
                    return (v / n).astype(np.float32) if n != 0 else None

                def embedding_from_landmarks(img, lmk68):
                    if lmk68 is None or len(lmk68) != 68:
                        return None
                    emb = wrapper.extract_embedding(img, lmk68)
                    if emb is None:
                        return None
                    return normalize_vec(emb)

                def collect_embeddings(search_dir):
                    sum_vec = None
                    count = 0
                    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                    for p in sorted(search_dir.rglob("*")):
                        if not p.is_file() or p.suffix.lower() not in exts:
                            continue
                        dflimg = DFLJPG.load(str(p))
                        if dflimg is not None and dflimg.has_data():
                            try:
                                landmarks = dflimg.get_landmarks()
                            except Exception:
                                landmarks = None
                            if landmarks is not None and len(landmarks) == 68:
                                img = read_image(p)
                                if img is None:
                                    continue
                                emb = embedding_from_landmarks(img, landmarks)
                                if emb is None:
                                    continue
                                if sum_vec is None:
                                    sum_vec = emb.copy()
                                else:
                                    sum_vec += emb
                                count += 1
                                continue
                        img = read_image(p)
                        if img is None:
                            continue
                        faces = wrapper.process_image(p, face_type_int)
                        if not faces:
                            continue
                        best_face = None
                        best_area = 0.0
                        for face in faces:
                            rect = face['rect']
                            area = max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])
                            if area > best_area:
                                best_area = area
                                best_face = face
                        if best_face is None:
                            continue
                        emb = best_face.get("embedding")
                        if emb is not None:
                            emb = normalize_vec(emb)
                        if emb is None:
                            emb = embedding_from_landmarks(img, best_face['landmarks'])
                        if emb is None:
                            continue
                        if sum_vec is None:
                            sum_vec = emb.copy()
                        else:
                            sum_vec += emb
                        count += 1
                    if count <= 0 or sum_vec is None:
                        return None, 0
                    mean_emb = sum_vec / max(1, count)
                    return normalize_vec(mean_emb), count

                ref_dir = Path(args.reference_dir)
                if not ref_dir.exists():
                    print(f"参考目录不存在: {ref_dir}")
                else:
                    subdirs = [d for d in ref_dir.iterdir() if d.is_dir()]
                    if args.multi_target and subdirs:
                        for sd in subdirs:
                            emb, count = collect_embeddings(sd)
                            if emb is not None and count > 0:
                                reference_embs.append(emb)
                                reference_names.append(sd.name)
                                reference_dirs.append(sd)
                        if reference_embs:
                            multi_target_active = True
                            print(f"参考人物数量: {len(reference_embs)}，阈值: {args.sim_threshold}")
                    if not multi_target_active:
                        emb, count = collect_embeddings(ref_dir)
                        if emb is not None and count > 0:
                            reference_emb = emb
                            print(f"参考人脸数量: {count}，阈值: {args.sim_threshold}")
                if multi_target_active and reference_embs:
                    target_backend_ready = wrapper.set_reference_embeddings(np.stack(reference_embs, axis=0).astype(np.float32), args.sim_threshold)
                elif reference_emb is not None:
                    target_backend_ready = wrapper.set_reference_embeddings(reference_emb, args.sim_threshold)
                else:
                    if hasattr(wrapper, "clear_reference_embeddings"):
                        wrapper.clear_reference_embeddings()
        if not target_backend_ready:
            print("指定人物分类失败: DLL 不支持目标分类或参考库为空")
            return
    else:
        if hasattr(wrapper, "clear_reference_embeddings"):
            wrapper.clear_reference_embeddings()

    if multi_target_active:
        target_output_dirs = [output_base / f"{name}_aligned" for name in reference_names]
        for d in target_output_dirs:
            if is_batch:
                ensure_dir(d)
            else:
                ensure_empty_dir(d)
        if args.keep_non_target:
            non_target_dir = output_base / "非目标人物_aligned"
            if is_batch:
                ensure_dir(non_target_dir)
            else:
                ensure_empty_dir(non_target_dir)
    else:
        if is_batch:
            ensure_dir(output_path)
        else:
            ensure_empty_dir(output_path)
        if args.keep_non_target:
            non_target_dir = output_base / "非目标人物_aligned"
            if is_batch:
                ensure_dir(non_target_dir)
            else:
                ensure_empty_dir(non_target_dir)

    try:
        non_target_keep_interval = int(getattr(args, "non_target_keep_interval", 5) or 5)
    except Exception:
        non_target_keep_interval = 5
    non_target_keep_interval = max(1, non_target_keep_interval)
    non_target_seen = {"count": 0}
    state_lock = threading.Lock()

    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"人脸类型: {face_type_str}")
    print(f"输出分辨率: {args.image_size}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.ts', '.m4v']
    input_is_video = input_path.is_file() and input_path.suffix.lower() in video_extensions
    input_is_image = input_path.is_file() and input_path.suffix.lower() in image_extensions
    if input_is_video:
        image_files = []
    elif input_is_image:
        image_files = [input_path]
    else:
        image_files = [p for p in input_path.glob('*') if p.suffix.lower() in image_extensions]
    total_items = len(image_files)
    cap = None
    fps = 0.0
    total_frames = 0
    skip_start = 0.0
    skip_end = 0.0
    frame_step = 1
    start_frame = 0
    end_frame = 0
    if input_is_video:
        if not hasattr(wrapper, "get_video_info") or not hasattr(wrapper, "read_video_frame"):
            print("当前 DLL 缺少视频接口（GetVideoInfo/ReadVideoFrame），前端仅支持 C++ 视频解码，已中止")
            return
        info = wrapper.get_video_info(input_path)
        if not info:
            print(f"无法通过 C++ 打开视频: {input_path}，请检查 DLL 编译选项与视频后端")
            return
        fps = float(info.get("fps", 0.0) or 0.0)
        total_frames = int(info.get("frame_count", 0) or 0)
        skip_start = float(getattr(args, "skip_start", 0.0) or 0.0)
        skip_end = float(getattr(args, "skip_end", 0.0) or 0.0)
        frame_step = int(getattr(args, "frame_step", 1) or 1)
        if frame_step <= 0:
            frame_step = 1
        start_frame = int(max(0.0, skip_start) * fps) if fps > 0 else 0
        end_frame = total_frames - int(max(0.0, skip_end) * fps) if fps > 0 and total_frames > 0 else total_frames
        if total_frames > 0:
            total_items = max(0, ((end_frame - start_frame) + frame_step - 1) // frame_step)
    
    print(f"待处理数量: {total_items}")
    
    pbar = tqdm(total=total_items, ncols=100, ascii=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    pbar_lock = threading.Lock()

    def update_pbar():
        with pbar_lock:
            pbar.update(1)

    def write_bytes(path, data):
        with open(path, "wb") as f:
            f.write(data)

    saver_workers = max(1, int(getattr(args, "saver_threads", 1) or 1))
    target_fast_mode = bool(args.reference_dir)
    keep_non_target = getattr(args, "keep_non_target", False)
    last_target_lock = threading.Lock()
    last_target_state = {"rect": None}

    def get_last_target_rect():
        with last_target_lock:
            return last_target_state["rect"]

    def set_last_target_rect(rect):
        with last_target_lock:
            last_target_state["rect"] = rect

    def iter_faces_with_hint(face_items, hint_rect):
        if not hint_rect:
            return face_items
        hx = (hint_rect[0] + hint_rect[2]) * 0.5
        hy = (hint_rect[1] + hint_rect[3]) * 0.5
        hw = max(1.0, hint_rect[2] - hint_rect[0])
        hh = max(1.0, hint_rect[3] - hint_rect[1])
        base_r = max(hw, hh) * 1.25
        tiers = [base_r, base_r * 2.0, base_r * 3.5, float("inf")]
        scored = []
        for idx, face in face_items:
            rect = face.get("rect")
            if not rect:
                dist = float("inf")
            else:
                cx = (rect[0] + rect[2]) * 0.5
                cy = (rect[1] + rect[3]) * 0.5
                dx = cx - hx
                dy = cy - hy
                dist = (dx * dx + dy * dy) ** 0.5
            scored.append((dist, idx, face))
        scored.sort(key=lambda x: x[0])
        yielded = set()
        ordered = []
        for r in tiers:
            for dist, idx, face in scored:
                if idx in yielded:
                    continue
                if dist <= r:
                    ordered.append((idx, face))
                    yielded.add(idx)
        return ordered

    # Worker function
    def process_worker(file_path=None, frame=None, source_name=None, source_stem=None):
        try:
            image = None
            source_image = None
            if frame is not None:
                if multi_target_active or reference_emb is not None:
                    image = frame
                faces = wrapper.process_frame(frame, face_type_int)
                source_image = frame
                if source_name is None:
                    source_name = "frame"
                if source_stem is None:
                    source_stem = Path(source_name).stem
            else:
                image = read_image(file_path)
                if image is None:
                    faces = wrapper.process_image(file_path, face_type_int)
                else:
                    faces = wrapper.process_frame(image, face_type_int)
                source_name = file_path.name
                source_stem = file_path.stem
                if args.classify_pose or args.debug:
                    source_image = image if image is not None else read_image(file_path)
            
            if not faces and args.save_no_face:
                if frame is None:
                    no_face_dir = file_path.parent / "未提取"
                    no_face_dir.mkdir(parents=True, exist_ok=True)
                    saver_pool.submit(shutil.copy, file_path, no_face_dir / file_path.name)
                else:
                    no_face_dir = input_path.parent / "未提取"
                    no_face_dir.mkdir(parents=True, exist_ok=True)
                    out_name = f"{source_stem}.png"
                    out_path = no_face_dir / out_name
                    ok, buf = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    if ok:
                        saver_pool.submit(write_bytes, out_path, buf.tobytes())
                return

            target_found_for_image = False
            face_items = list(enumerate(faces))
            if target_fast_mode:
                hint_rect = get_last_target_rect()
                face_items = iter_faces_with_hint(face_items, hint_rect)
            for idx, face in face_items:
                base_out_dir = output_path
                emb = None
                if multi_target_active or reference_emb is not None:
                    if target_found_for_image:
                        if not keep_non_target or non_target_dir is None:
                            continue
                        if getattr(args, "reduce_non_target", False):
                            with state_lock:
                                non_target_seen["count"] += 1
                                if non_target_seen["count"] % non_target_keep_interval != 0:
                                    continue
                        base_out_dir = non_target_dir
                    else:
                        is_target = face.get("is_target")
                        target_idx = face.get("target_index", -1)
                        if is_target is None:
                            print("指定人物分类失败: 未从 DLL 返回分类结果")
                            continue
                        if not is_target:
                            if not keep_non_target:
                                continue
                            if non_target_dir is None:
                                continue
                            if getattr(args, "reduce_non_target", False):
                                with state_lock:
                                    non_target_seen["count"] += 1
                                    if non_target_seen["count"] % non_target_keep_interval != 0:
                                        continue
                            base_out_dir = non_target_dir
                        else:
                            target_found_for_image = True
                            if multi_target_active and target_idx is not None and target_idx >= 0 and target_idx < len(target_output_dirs):
                                base_out_dir = target_output_dirs[target_idx]
                            else:
                                base_out_dir = output_path

                output_file = None
                try:
                    jpg_bytes = face['jpg_data']
                    decoded = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is None:
                        continue
                    aligned_landmarks = face['aligned_landmarks']
                    source_landmarks = np.array(face['landmarks'], dtype=np.float32)
                    
                    final_out_dir = base_out_dir
                    skip_sub_filters = False
                    if args.blur_classify:
                        variance = face.get("blur")
                        if variance is None or variance <= 0:
                            print("清晰度分类失败: 未从 DLL 返回结果")
                            continue
                        if variance is None:
                            tag = "blur_error"
                        elif variance < blur_low:
                            tag = "低方差"
                        elif variance < blur_high:
                            tag = "中方差"
                        else:
                            tag = "高方差"
                        final_out_dir = final_out_dir / tag
                        if tag == "高方差":
                            skip_sub_filters = True

                    if not skip_sub_filters and args.classify_pose:
                        pose_tag = face.get('pose')
                        if not pose_tag:
                            print("姿态分类失败: 未从 DLL 返回结果")
                            continue
                        if args.target_pose_type:
                            if pose_tag == args.target_pose_type:
                                final_out_dir = final_out_dir / pose_tag
                        else:
                            final_out_dir = final_out_dir / pose_tag

                    if not skip_sub_filters and args.select_mouth_open:
                        if "mouth_open" not in face:
                            print("张嘴分类失败: 未从 DLL 返回结果")
                            continue
                        mouth_open = face.get("mouth_open")
                        if mouth_open:
                            final_out_dir = final_out_dir / "张嘴"

                    final_out_dir.mkdir(parents=True, exist_ok=True)
                    output_filename = f"{source_stem}_{idx}.jpg"
                    output_file = final_out_dir / output_filename
                    dflimg = DFLJPG.load_from_memory(jpg_bytes, str(output_file))
                    dflimg.set_face_type(face_type_str)
                    dflimg.set_landmarks(aligned_landmarks)
                    dflimg.set_source_filename(source_name)
                    source_rect = np.array([float(x) for x in face['rect']], dtype=np.float32)
                    dflimg.set_source_rect(source_rect)
                    dflimg.set_source_landmarks(source_landmarks.tolist())
                    try:
                        face_type_enum = FaceType.fromString(face_type_str)
                        face_mat = LandmarksProcessor.get_transform_mat(source_landmarks, decoded.shape[0], face_type_enum)
                        dflimg.set_image_to_face_mat(face_mat)
                    except Exception:
                        pass
                    saver_pool.submit(write_bytes, output_file, dflimg.dump())
                    if target_fast_mode and target_found_for_image:
                        set_last_target_rect(face.get("rect"))
                        if not keep_non_target:
                            break

                    if args.debug and source_image is not None:
                        dbg = source_image.copy()
                        try:
                            LandmarksProcessor.draw_landmarks(dbg, source_landmarks, transparent_mask=True)
                            x1, y1, x2, y2 = [int(v) for v in face['rect']]
                            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        except Exception:
                            pass
                        debug_file = debug_path / f"{source_stem}_{idx}_debug.jpg"
                        ok, dbg_buf = cv2.imencode(".jpg", dbg)
                        if ok:
                            saver_pool.submit(write_bytes, debug_file, dbg_buf.tobytes())
                        
                except Exception as e:
                    print(f"Error saving {output_file}: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
        finally:
            update_pbar()

    with concurrent.futures.ThreadPoolExecutor(max_workers=saver_workers) as saver_pool:
        if input_is_video:
            num_workers = max(1, args.loader_threads)
            print(f"视频处理模式: 并发线程数 {num_workers}")
            
            if end_frame > 0 and end_frame <= start_frame:
                print("视频有效帧数为 0")
                pbar.close()
                return

            # Producer-Consumer for video
            # Use multiple readers to saturate bandwidth/CPU decoding
            num_readers = min(4, num_workers)
            print(f"启动 {num_readers} 个视频读取线程")
            
            frame_queue = queue.Queue(maxsize=num_workers * 3)
            stop_read = threading.Event()
            
            total_range = end_frame - start_frame
            if total_range > 0:
                chunk_size = math.ceil(total_range / num_readers)
            else:
                num_readers = 1
                chunk_size = 0
            
            def video_reader(reader_id, chunk_start, chunk_end):
                try:
                    cap = cv2.VideoCapture(str(input_path))
                    if not cap.isOpened():
                        print(f"Reader {reader_id}: 无法打开视频: {input_path}")
                        return
                    
                    # Seek
                    if chunk_start > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
                    
                    current_frame = chunk_start
                    while not stop_read.is_set():
                        if chunk_end > 0 and current_frame >= chunk_end:
                            break
                            
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if (current_frame - start_frame) % frame_step == 0:
                            source_name = f"{input_path.name}@{current_frame:06d}"
                            source_stem = f"{input_path.stem}_{current_frame:06d}"
                            while not stop_read.is_set():
                                try:
                                    frame_queue.put((frame, source_name, source_stem), timeout=1)
                                    break
                                except queue.Full:
                                    continue
                        
                        current_frame += 1
                    cap.release()
                except Exception as e:
                    print(f"Reader {reader_id} error: {e}")

            reader_threads = []
            for i in range(num_readers):
                if chunk_size > 0:
                    cs = start_frame + i * chunk_size
                    ce = min(start_frame + (i + 1) * chunk_size, end_frame)
                    if cs >= ce:
                        continue
                else:
                    cs = start_frame
                    ce = 0
                    
                t = threading.Thread(target=video_reader, args=(i, cs, ce), daemon=True)
                t.start()
                reader_threads.append(t)

            def wait_readers_and_finalize():
                for t in reader_threads:
                    t.join()
                # Put sentinels for all workers
                for _ in range(num_workers):
                    frame_queue.put(None)

            # Start a thread to wait for readers and put sentinels
            finalize_thread = threading.Thread(target=wait_readers_and_finalize, daemon=True)
            finalize_thread.start()

            def video_worker_loop():
                while True:
                    item = frame_queue.get()
                    if item is None:
                        # Sentinel received. 
                        # Note: We put one sentinel per worker, so we don't need to propagate.
                        # But if we did propagate, it would be: frame_queue.put(None)
                        frame_queue.task_done()
                        break
                    
                    frame, source_name, source_stem = item
                    try:
                        process_worker(frame=frame, source_name=source_name, source_stem=source_stem)
                    except Exception as e:
                        print(f"Worker error: {e}")
                    finally:
                        frame_queue.task_done()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(video_worker_loop) for _ in range(num_workers)]
                # Wait for all
                for f in concurrent.futures.as_completed(futures):
                    try:
                        f.result()
                    except Exception:
                        pass
            
            stop_read.set()
            finalize_thread.join()

        else:
            num_workers = max(1, args.loader_threads)
            print(f"图片处理模式: 并发线程数 {num_workers}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_worker, p) for p in image_files]
                for _ in concurrent.futures.as_completed(futures):
                    pass

    pbar.close()
    elapsed = time.perf_counter() - start_time
    print(f"Run time: {elapsed:.2f}s")
    print("Done!")
