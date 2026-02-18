"""
测试单个模型
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir / "sdk" / "_libs"))

from FaceExtractorWrapper import FaceExtractorWrapper

test_image = Path(r"D:\Program Files\face_classification\TensorFlow_Extract_Revision\data\output\cpp版本\S3FD+2DFAN\00100_0.jpg")

if not test_image.exists():
    print(f"测试图片不存在: {test_image}")
    sys.exit(1)

models_dir = root_dir / "assets" / "models"

print("测试 1k3d68 模型")
wrapper = FaceExtractorWrapper(
    str(models_dir),
    device_id=0,
    fan_model_path=str(models_dir / "1k3d68.onnx")
)
faces = wrapper.process_image(str(test_image), face_type=2)
print(f"结果: 检测到 {len(faces)} 个人脸")
