import sys
from pathlib import Path
import numpy as np
import cv2

current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
libs_dir = root_dir / "sdk" / "_libs"
sys.path.insert(0, str(libs_dir))

try:
    from DFLIMG import DFLJPG
except Exception:
    sys.path.insert(0, str(libs_dir / "facelib"))
    from DFLIMG import DFLJPG
from facelib import LandmarksProcessor

def extract_insight_landmarks(img_bgr, source_rect, model_path, input_size=192):
    import onnxruntime as ort
    x1, y1, x2, y2 = source_rect
    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    size = max(w, h)
    s = float(input_size) / (size * 1.5)
    half = float(input_size) * 0.5
    M = np.array([[s, 0.0, half - cx * s], [0.0, s, half - cy * s]], dtype=np.float32)
    crop = cv2.warpAffine(img_bgr, M, (input_size, input_size), flags=cv2.INTER_LINEAR)
    M_inv = cv2.invertAffineTransform(M).astype(np.float64)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = np.transpose(rgb, (2, 0, 1))[None, :, :, :]
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: blob})[0]
    v = np.array(out).reshape(-1)
    if v.size >= 3000 and v.size % 3 == 0:
        vtx = v.size // 3
        pc = 68
        off = (vtx - pc) * 3
        pts = v[off:off + pc * 3].reshape(pc, 3)[:, :2]
    elif v.size == 212:
        pts = v.reshape(106, 2)
    elif v.size == 136:
        pts = v.reshape(68, 2)
    elif v.size == 204:
        pts = v.reshape(68, 3)[:, :2]
    elif v.size % 3 == 0 and v.size // 3 == 68:
        pts = v.reshape(68, 3)[:, :2]
    elif v.size % 2 == 0:
        pts = v.reshape(v.size // 2, 2)
    else:
        return None
    pts = (pts + 1.0) * half
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
    out = (M_inv @ pts_h.T).T
    return out[:, :2]

def read_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def save_image(path, img):
    try:
        ext = path.suffix if path.suffix else ".jpg"
        buf = cv2.imencode(ext, img)[1]
        buf.tofile(str(path))
        return True
    except Exception:
        return False

def draw_points(image, landmarks, color=(0, 255, 0), radius=3, thickness=2, draw_index=True, font_scale=0.4, font_thickness=1):
    for idx, (x, y) in enumerate(landmarks):
        cx = int(x)
        cy = int(y)
        cv2.circle(image, (cx, cy), radius, color, thickness, lineType=cv2.LINE_AA)
        if draw_index:
            cv2.putText(image, str(idx), (cx + radius + 1, cy - radius - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

def normalize_path_input(text):
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    quotes = ['"', "'", "“", "”", "‘", "’", "''", "‘‘", "’’", "``"]
    if len(s) >= 2:
        for q in quotes:
            if s.startswith(q) and s.endswith(q):
                s = s[len(q):-len(q)]
                break
    return s.strip()

def process_path(img_path, mode="dfl", model_path=None):
    img = read_image(img_path)
    if img is None:
        print(f"读取失败: {img_path}")
        return
    dflimg = None
    try:
        dflimg = DFLJPG.load(str(img_path))
    except Exception:
        dflimg = None
    if mode == "dfl":
        if dflimg is None or not dflimg.has_data():
            print("未读取到 DFL 数据")
            return
    landmarks = None
    if mode == "dfl":
        landmarks = dflimg.get_landmarks()
        if landmarks is None or len(landmarks) != 68:
            landmarks = dflimg.get_source_landmarks()
        if landmarks is None or len(landmarks) != 68:
            print("未读取到 68 点 landmarks")
            return
    else:
        rect = None
        if dflimg is not None and dflimg.has_data():
            rect = dflimg.get_source_rect()
        if rect is None or len(rect) != 4:
            try:
                from FaceExtractorWrapper import FaceExtractorWrapper
                model_dir = root_dir / "assets" / "models"
                extractor = FaceExtractorWrapper(model_dir, device_id=-1)
                faces = extractor.process_image(img_path, face_type=2)
                rect = faces[0]["rect"] if faces else None
            except Exception:
                rect = None
        if rect is None or len(rect) != 4:
            print("未获取到人脸框")
            return
        if model_path is None:
            if mode == "2d106det":
                model_path = root_dir / "assets" / "models" / "2d106det.onnx"
            elif mode == "1k3d68":
                model_path = root_dir / "assets" / "models" / "1k3d68.onnx"
            else:
                print("未知模式")
                return
        if not Path(model_path).exists():
            print(f"模型不存在: {model_path}")
            return
        landmarks = extract_insight_landmarks(img, rect, model_path)
        if landmarks is None:
            print("模型输出解析失败")
            return
    out_circles = img.copy()
    out_landmarks = img.copy()
    out_points = img.copy()
    try:
        LandmarksProcessor.draw_landmarks(out_circles, landmarks, transparent_mask=False, draw_circles=True)
    except Exception:
        pass
    try:
        LandmarksProcessor.draw_landmarks(out_landmarks, landmarks, transparent_mask=True, draw_circles=False)
    except Exception:
        pass
    try:
        draw_points(out_points, landmarks, color=(0, 255, 0), radius=3, thickness=2, draw_index=True, font_scale=0.4, font_thickness=1)
    except Exception:
        pass
    out_dir = img_path.parent
    base = img_path.stem
    tag = "" if mode == "dfl" else f"_{mode}"
    outputs = [
        (out_dir / f"{base}{tag}_circles.jpg", out_circles),
        (out_dir / f"{base}{tag}_landmarks.jpg", out_landmarks),
        (out_dir / f"{base}{tag}_points.jpg", out_points),
    ]
    for out_path, out_img in outputs:
        if save_image(out_path, out_img):
            print(f"已输出: {out_path}")
        else:
            print(f"输出失败: {out_path}")

def main():
    mode = "dfl"
    model_path = None
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if len(args) >= 2 and args[0] == "--mode":
            mode = normalize_path_input(args[1]).lower()
            args = args[2:]
        if len(args) >= 2 and args[0] == "--model":
            model_path = Path(normalize_path_input(args[1]))
            args = args[2:]
        img_path = Path(normalize_path_input(" ".join(args)))
        if not img_path.exists():
            print(f"文件不存在: {img_path}")
            return
        process_path(img_path, mode=mode, model_path=model_path)
        return
    while True:
        user_input = normalize_path_input(input("请输入图片路径(回车退出): "))
        if not user_input:
            return
        img_path = Path(user_input)
        if not img_path.exists():
            print(f"文件不存在: {img_path}")
            continue
        process_path(img_path, mode=mode, model_path=model_path)

if __name__ == "__main__":
    main()
