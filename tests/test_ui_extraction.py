"""
测试 UI 是否真的能在已提取的图片上再次提取
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir / "sdk" / "_libs"))

from FaceExtractorWrapper import FaceExtractorWrapper

def test_ui_extraction():
    test_image = Path(r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\data\output\cpp版本\S3FD+2DFAN\00100_0.jpg")
    
    if not test_image.exists():
        print(f"测试图片不存在: {test_image}")
        return
    
    models_dir = root_dir / "assets" / "models"
    
    print("=" * 60)
    print("测试 UI 提取流程")
    print("=" * 60)
    print(f"测试图片: {test_image}")
    print()
    
    # 使用默认模型（S3FD + 2DFAN）
    print("[1] 测试默认模型 (S3FD + 2DFAN)")
    try:
        wrapper = FaceExtractorWrapper(str(models_dir), device_id=0)
        faces = wrapper.process_image(str(test_image), face_type=2)
        print(f"   结果: 检测到 {len(faces)} 个人脸")
        if faces:
            print(f"   成功提取")
        else:
            print(f"   未检测到人脸")
        del wrapper  # 显式删除以释放资源
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 使用 1k3d68
    print("[2] 测试 1k3d68 模型")
    try:
        wrapper = FaceExtractorWrapper(
            str(models_dir),
            device_id=0,
            fan_model_path=str(models_dir / "1k3d68.onnx")
        )
        faces = wrapper.process_image(str(test_image), face_type=2)
        print(f"   结果: 检测到 {len(faces)} 个人脸")
        if faces:
            print(f"   成功提取")
        else:
            print(f"   未检测到人脸")
        del wrapper  # 显式删除以释放资源
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 使用 2d106det
    print("[3] 测试 2d106det 模型")
    try:
        wrapper = FaceExtractorWrapper(
            str(models_dir),
            device_id=0,
            fan_model_path=str(models_dir / "2d106det.onnx")
        )
        faces = wrapper.process_image(str(test_image), face_type=2)
        print(f"   结果: 检测到 {len(faces)} 个人脸")
        if faces:
            print(f"   成功提取")
        else:
            print(f"   未检测到人脸")
        del wrapper  # 显式删除以释放资源
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_ui_extraction()
