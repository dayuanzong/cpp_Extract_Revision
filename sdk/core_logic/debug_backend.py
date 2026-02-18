from pathlib import Path
import sys

def get_landmarks_from_image(img_path):
    try:
        libs_dir = Path(__file__).parent.parent / "_libs"
        libs_path = str(libs_dir)
        if libs_path not in sys.path:
            sys.path.insert(0, libs_path)
        from DFLIMG import DFLJPG
        dflimg = DFLJPG.load(str(img_path))
        if dflimg and dflimg.has_data():
            landmarks = dflimg.get_landmarks()
            if landmarks is None or len(landmarks) != 68:
                landmarks = dflimg.get_source_landmarks()
            if landmarks is not None and len(landmarks) == 68:
                return landmarks
    except Exception:
        return None
    return None

def draw_points(image, landmarks, color=(0, 255, 0), radius=3, thickness=2, draw_index=True, font_scale=0.4, font_thickness=1):
    import cv2
    for idx, (x, y) in enumerate(landmarks):
        cx = int(x)
        cy = int(y)
        cv2.circle(image, (cx, cy), radius, color, thickness, lineType=cv2.LINE_AA)
        if draw_index:
            cv2.putText(image, str(idx), (cx + radius + 1, cy - radius - 1), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, 
                        lineType=cv2.LINE_AA)

def load_thumbnail_turbojpeg(jpeg_decoder, img_file, thumbnail_size):
    try:
        import tkinter as tk
        import numpy as np
        with open(img_file, 'rb') as f:
            jpeg_data = f.read()
        bgr_array = jpeg_decoder.decode(jpeg_data, scaling_factor=(1, 8))
        rgb_array = bgr_array[:, :, ::-1]
        h, w = rgb_array.shape[:2]
        scale = min(thumbnail_size / h, thumbnail_size / w)
        if scale < 1:
            new_h = int(h * scale)
            new_w = int(w * scale)
            if new_h <= 0 or new_w <= 0:
                return None
            indices_h = np.linspace(0, h - 1, new_h).astype(int)
            indices_w = np.linspace(0, w - 1, new_w).astype(int)
            rgb_array = rgb_array[indices_h, :, :][:, indices_w, :]
        h, w = rgb_array.shape[:2]
        ppm_header = f'P6 {w} {h} 255 '.encode()
        ppm_data = ppm_header + rgb_array.tobytes()
        photo = tk.PhotoImage(data=ppm_data, format='PPM')
        return photo
    except Exception:
        return None
