import sys
from pathlib import Path
import numpy as np
import cv2
import time

"""
C++ 版本特征点提取精度优化示例
------------------------------
C++ 版 (cpp_Extract_Revision) 设计初衷是为了追求极高的推理速度，因此默认只执行单次检测和特征点提取 (Single Pass)。
相比之下，Python 原版 (TensorFlow_Extract) 为了保证精度，默认执行了二次提取 (Second Pass)：
1. 初步检测人脸。
2. 将人脸对齐并裁剪为 256x256。
3. 在裁剪图上再次运行检测器 (S3FD)。
4. 在裁剪图上运行特征点提取 (FAN)，甚至进行多重采样 (Multi-sample)。

这种差异导致 C++ 版虽然速度极快，但在某些大角度或遮挡情况下，特征点精度不如 Python 原版。

本脚本演示了如何在 Python 中利用 C++ DLL 实现类似的“二次提取”逻辑，从而在不修改 C++ 源码的情况下显著提升精度。
"""

current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
libs_dir = root_dir / "sdk" / "_libs"
sys.path.insert(0, str(libs_dir))

from FaceExtractorWrapper import FaceExtractorWrapper
try:
    from facelib import LandmarksProcessor, FaceType
except ImportError:
    # 尝试从原代码目录导入 LandmarksProcessor 用于对齐计算
    ref_dir = root_dir.parent / "原代码提供参考"
    sys.path.insert(0, str(ref_dir))
    from facelib import LandmarksProcessor, FaceType

def transform_points(points, mat, invert=False):
    """
    对点集应用仿射变换矩阵
    """
    points = np.expand_dims(points, axis=1)
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = cv2.transform(points, mat)
    return points.squeeze()

def extract_accurate(wrapper, img):
    """
    执行高精度提取流程：Pass 1 -> Align -> Pass 2 -> Transform Back
    """
    # Pass 1: 初步提取
    faces = wrapper.process_frame(img)
    if not faces:
        return []
    
    results = []
    for face in faces:
        lm_pass1 = np.array(face["landmarks"], dtype=np.float32)
        rect_pass1 = face["rect"]
        
        # Pass 2: 二次提取
        try:
            # 1. 计算对齐矩阵 (256x256, Full Face)
            mat = LandmarksProcessor.get_transform_mat(lm_pass1, 256, FaceType.FULL)
            
            # 2. 变换图像
            img_aligned = cv2.warpAffine(img, mat, (256, 256), flags=cv2.INTER_CUBIC)
            
            # 3. 在对齐图上再次检测
            # 注意：这里我们利用 DLL 的检测能力，在对齐后的清晰人脸上再次寻找特征点
            faces_aligned = wrapper.process_frame(img_aligned)
            
            if faces_aligned:
                # 找到最大的人脸（通常对齐图里只有一张脸，且占满画面）
                best_face = max(faces_aligned, key=lambda f: (f["rect"][2]-f["rect"][0])*(f["rect"][3]-f["rect"][1]))
                lm_aligned = np.array(best_face["landmarks"], dtype=np.float32)
                
                # 4. 变换回原图坐标
                lm_pass2 = transform_points(lm_aligned, mat, invert=True)
                
                # 更新 face 数据
                face["landmarks"] = lm_pass2.tolist()
                # 注意：aligned_landmarks 是 256x256 下的，这里其实就是 lm_aligned
                face["aligned_landmarks"] = lm_aligned.tolist()
                
        except Exception as e:
            print(f"二次提取失败: {e}，回退到 Pass 1 结果")
            
        results.append(face)
        
    return results

def main():
    # 1. 初始化
    model_dir = root_dir / "assets" / "models"
    if not model_dir.exists():
        print(f"模型目录不存在: {model_dir}")
        return
        
    wrapper = FaceExtractorWrapper(model_dir, 0)
    
    # 2. 读取测试图片
    input_dir = root_dir.parent / "data" / "input" / "包含面部图片"
    img_path = None
    if input_dir.exists():
        files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        if files:
            img_path = files[0]
            
    if not img_path:
        print("未找到测试图片，请检查 data/input/包含面部图片")
        return

    print(f"测试图片: {img_path}")
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # 3. 运行高精度提取
    print("开始高精度提取...")
    t0 = time.time()
    faces = extract_accurate(wrapper, img)
    t1 = time.time()
    print(f"耗时: {t1-t0:.4f}s")
    
    if faces:
        print(f"成功提取 {len(faces)} 张人脸")
        # 可视化结果
        out = img.copy()
        for face in faces:
            lm = np.array(face["landmarks"], dtype=np.int32)
            for x, y in lm:
                cv2.circle(out, (x, y), 2, (0, 255, 0), -1)
                
        out_path = root_dir.parent / "data" / "output" / "_debug_compare" / f"{img_path.stem}_accurate.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", out)[1].tofile(str(out_path))
        print(f"结果已保存: {out_path}")

if __name__ == "__main__":
    main()
