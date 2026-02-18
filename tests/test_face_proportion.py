"""
测试不同模型提取的面部比例是否一致
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 添加路径
root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir / "sdk" / "_libs"))

from FaceExtractorWrapper import FaceExtractorWrapper

def test_face_proportion():
    """测试不同模型的面部比例"""
    
    # 使用用户指定的测试图片（256x256已提取的人脸）
    test_image = Path(r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\data\output\cpp版本\S3FD+2DFAN\00100_0.jpg")
    
    if not test_image.exists():
        print(f"❌ 测试图片不存在: {test_image}")
        return
    
    # 检查图片尺寸 - 使用 cv2.imdecode 处理中文路径
    try:
        with open(test_image, 'rb') as f:
            img_data = f.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 无法读取测试图片: {test_image}")
        print(f"   错误: {e}")
        return
    
    if img is None:
        print(f"❌ 无法解码测试图片: {test_image}")
        return
    
    h, w = img.shape[:2]
    print(f"使用测试图片: {test_image}")
    print(f"图片尺寸: {w}x{h}")
    
    # 对于小图片（已提取的人脸），S3FD检测器可能无法检测到人脸
    # 这是正常的，因为S3FD是为大图设计的
    # UI可以成功提取是因为它使用了不同的处理流程
    if max(h, w) < 640:
        print(f"⚠ 警告: 图片尺寸较小 ({w}x{h})")
        print(f"   如果测试失败，请使用原始大图（至少640px）进行测试")
    
    models_dir = root_dir / "assets" / "models"
    if not models_dir.exists():
        print(f"模型目录不存在: {models_dir}")
        return
    
    print("=" * 60)
    print("面部比例一致性测试")
    print("=" * 60)
    
    # 测试配置
    test_configs = [
        {
            "name": "2DFAN",
            "fan_path": "FAN/2DFAN.onnx"
        },
        {
            "name": "1k3d68",
            "fan_path": "1k3d68.onnx"
        },
        {
            "name": "2d106det",
            "fan_path": "2d106det.onnx"
        }
    ]
    
    results = {}
    
    for config in test_configs:
        model_name = config["name"]
        fan_path = models_dir / config["fan_path"]
        
        if not fan_path.exists():
            print(f"\n[跳过] {model_name}: 模型文件不存在 {fan_path}")
            continue
        
        print(f"\n[测试] {model_name}")
        print(f"  模型路径: {fan_path}")
        
        try:
            # 初始化提取器
            wrapper = FaceExtractorWrapper(
                str(models_dir),
                device_id=0,
                fan_model_path=str(fan_path)
            )
            
            # 处理图片
            faces = wrapper.process_image(str(test_image), face_type=2)
            
            # 显式删除 wrapper 以释放资源，避免多个实例冲突
            del wrapper
            
            if not faces:
                print(f"  ❌ 未检测到人脸")
                continue
            
            # 获取第一个人脸
            face = faces[0]
            jpg_data = face["jpg_data"]
            
            # 解码图片
            nparr = np.frombuffer(jpg_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"  ❌ 无法解码图片")
                continue
            
            h, w = img.shape[:2]
            landmarks = face["landmarks"]
            
            # 计算面部特征尺寸
            # 眼睛中心
            left_eye = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
            right_eye = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
            eye_distance = np.linalg.norm(left_eye - right_eye)
            
            # 鼻子到下巴
            nose_tip = landmarks[30]
            chin = landmarks[8]
            face_height = np.linalg.norm(np.array(nose_tip) - np.array(chin))
            
            # 面部宽度（脸颊）
            left_cheek = landmarks[2]
            right_cheek = landmarks[14]
            face_width = np.linalg.norm(np.array(left_cheek) - np.array(right_cheek))
            
            # 图片文件大小
            jpg_size = len(jpg_data)
            
            results[model_name] = {
                "image_size": (w, h),
                "jpg_size": jpg_size,
                "eye_distance": eye_distance,
                "face_height": face_height,
                "face_width": face_width,
                "eye_distance_ratio": eye_distance / w,
                "face_height_ratio": face_height / h,
                "face_width_ratio": face_width / w
            }
            
            print(f"  ✓ 图片尺寸: {w}x{h}")
            print(f"  ✓ 文件大小: {jpg_size/1024:.1f} KB")
            print(f"  ✓ 眼距: {eye_distance:.2f}px ({eye_distance/w*100:.1f}%)")
            print(f"  ✓ 面部高度: {face_height:.2f}px ({face_height/h*100:.1f}%)")
            print(f"  ✓ 面部宽度: {face_width:.2f}px ({face_width/w*100:.1f}%)")
            
            # 保存测试结果
            output_path = root_dir / f"test_output_{model_name}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"  ✓ 保存到: {output_path}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 比较结果
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("比例一致性分析")
        print("=" * 60)
        
        model_names = list(results.keys())
        base_model = model_names[0]
        base_result = results[base_model]
        
        print(f"\n以 {base_model} 为基准:")
        
        for model_name in model_names[1:]:
            result = results[model_name]
            
            # 文件大小差异
            size_diff = abs(result["jpg_size"] - base_result["jpg_size"]) / base_result["jpg_size"]
            
            # 比例差异
            eye_diff = abs(result["eye_distance_ratio"] - base_result["eye_distance_ratio"])
            height_diff = abs(result["face_height_ratio"] - base_result["face_height_ratio"])
            width_diff = abs(result["face_width_ratio"] - base_result["face_width_ratio"])
            
            print(f"\n{model_name} vs {base_model}:")
            print(f"  文件大小差异: {size_diff*100:.2f}%")
            print(f"  眼距比例差异: {eye_diff*100:.2f}%")
            print(f"  面部高度比例差异: {height_diff*100:.2f}%")
            print(f"  面部宽度比例差异: {width_diff*100:.2f}%")
            
            # 判断是否一致（允许5%的误差）
            threshold = 0.05
            if eye_diff < threshold and height_diff < threshold and width_diff < threshold:
                print(f"  ✓ 比例一致（差异 < {threshold*100}%）")
            else:
                print(f"  ⚠ 比例存在差异（差异 >= {threshold*100}%）")
    elif len(results) == 0:
        print("\n" + "=" * 60)
        print("⚠ 无法完成测试")
        print("=" * 60)
        print("\n所有模型都未能检测到人脸")
        print("可能原因:")
        print("  1. 图片质量问题")
        print("  2. 需要使用原始大图（至少640px）")
        print("  3. 检查模型文件是否正确")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_face_proportion()
