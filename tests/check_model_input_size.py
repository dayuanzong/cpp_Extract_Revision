"""
检查ONNX模型的输入尺寸
"""
import onnxruntime as ort
from pathlib import Path

def check_model_input_size(model_path):
    """检查模型的输入尺寸"""
    try:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        # 获取输入信息
        input_info = session.get_inputs()[0]
        print(f"\n模型: {model_path.name}")
        print(f"  输入名称: {input_info.name}")
        print(f"  输入形状: {input_info.shape}")
        print(f"  输入类型: {input_info.type}")
        
        # 获取输出信息
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  输出{i}: {output_info.name}, 形状: {output_info.shape}")
        
        return input_info.shape
    except Exception as e:
        print(f"\n❌ 无法加载模型 {model_path.name}: {e}")
        return None

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    models_dir = root_dir / "assets" / "models"
    
    print("=" * 60)
    print("检查模型输入尺寸")
    print("=" * 60)
    
    # 检查所有模型
    models_to_check = [
        "FAN/2DFAN.onnx",
        "1k3d68.onnx",
        "2d106det.onnx",
        "S3FD/S3FD.onnx",
        "det_10g.onnx",
        "w600k_r50.onnx"
    ]
    
    for model_rel_path in models_to_check:
        model_path = models_dir / model_rel_path
        if model_path.exists():
            check_model_input_size(model_path)
        else:
            print(f"\n⚠ 模型不存在: {model_path}")
    
    print("\n" + "=" * 60)
