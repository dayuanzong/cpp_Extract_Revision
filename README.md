# Face Extraction C++ Revision

人脸提取 C++ 优化版本，包含 1k3d68 模型优化框架。

## 项目概述

本项目是人脸检测和关键点提取的 C++ 实现，支持多种模型：
- **S3FD / SCRFD**: 人脸检测
- **1k3d68**: 68点3D人脸关键点检测（已优化）
- **2d106det**: 106点2D人脸关键点检测
- **2DFAN**: 68点2D人脸关键点检测（参考基准）
- **InsightFace**: 人脸识别

## 最新更新

### 2026-02-18: 1k3d68 优化框架

✅ **已完成**:
- 模型类型自动检测（1k3d68 / 2d106det）
- 独立配置管理系统
- 多采样支持（可配置）
- 输出验证机制
- 完整的测试套件

✅ **修复**:
- 修复了裁剪因子配置问题
- 恢复到稳定的默认配置

## 项目结构

```
cpp_Extract_Revision/
├── core/                           # C++ 核心代码
│   ├── src/
│   │   ├── InsightFaceLandmark.h   # 关键点检测（已优化）
│   │   ├── InsightFaceLandmark.cpp
│   │   ├── FANExtractor.h          # 2DFAN 实现
│   │   ├── FANExtractor.cpp
│   │   ├── FacePipeline.h          # 主流程
│   │   └── FacePipeline.cpp
│   └── build_cpp.bat               # 编译脚本
├── bin/                            # 编译输出（DLL）
├── sdk/                            # Python SDK
│   └── _libs/
│       └── FaceExtractorWrapper.py # Python 接口
├── tests/                          # 测试脚本
│   ├── test_phase1_basic.py        # 基础功能测试
│   ├── test_final_verification.py  # 最终验证
│   ├── test_compare_2dfan.py       # 与 2DFAN 对比
│   ├── experiment_*.py             # 实验脚本
│   ├── QUICK_START.md              # 快速启动指南
│   ├── EXPERIMENT_GUIDE.md         # 实验指南
│   └── CURRENT_STATUS.md           # 当前状态
├── assets/                         # 模型文件
│   └── models/
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── det_10g.onnx
│       └── FAN/
│           └── 2DFAN-4.onnx
└── .kiro/specs/1k3d68-optimization/ # 规格文档
    ├── requirements.md
    ├── design.md
    └── tasks.md
```

## 快速开始

### 1. 编译 C++ 代码

```bash
cd core
build_cpp.bat
```

### 2. 运行测试

```bash
cd tests
python test_final_verification.py
```

### 3. 使用 Python SDK

```python
from _libs.FaceExtractorWrapper import FaceExtractorWrapper

# 初始化
wrapper = FaceExtractorWrapper(
    "assets/models",
    device_id=-1,  # -1 for CPU, 0+ for GPU
    fan_model_path="assets/models/1k3d68.onnx"
)

# 处理图片
faces = wrapper.process_image("test.jpg", face_type=2)

# 获取结果
for face in faces:
    landmarks = face['landmarks']  # 关键点
    jpg_data = face['jpg_data']    # 对齐后的人脸图片
```

## 当前配置

### 1k3d68（已优化）
- **crop_factor**: 1.75f（与原来相同，稳定）
- **norm_mode**: AUTO（自动选择归一化模式）
- **multi_sample**: disabled（当前禁用，可启用）

### 2d106det（保持不变）
- **crop_factor**: 1.75f
- **norm_mode**: AUTO
- **multi_sample**: disabled

## 功能特性

### 模型类型检测
自动识别模型类型并应用对应配置：
```cpp
InsightFaceLandmark: Model type detected as 1K3D68
```

### 独立配置管理
每个模型有独立的配置，互不影响：
```cpp
struct ModelConfig {
    ModelType type;
    CropConfig crop;
    PreprocessConfig preprocess;
    MultiSampleConfig multi_sample;
};
```

### 多采样支持
可选的多采样功能提高稳定性：
- 5点采样（中心 + 上下左右）
- 9点采样（额外4个对角线）
- 自动平均结果

### 输出验证
自动验证输出质量：
- 检查关键点数量
- 检测 NaN/Inf 值
- 验证坐标范围

## 实验和优化

如果需要进一步优化精度，参考：

1. **快速启动**: `tests/QUICK_START.md`
2. **实验指南**: `tests/EXPERIMENT_GUIDE.md`
3. **当前状态**: `tests/CURRENT_STATUS.md`

### 实验脚本
- `experiment_crop_factor.py` - 裁剪因子优化
- `experiment_norm_mode.py` - 归一化模式优化
- `experiment_auto.py` - 自动化全面优化

## 依赖

### C++ 依赖
- Visual Studio 2019+
- CMake 3.15+
- ONNX Runtime 1.16+
- OpenCV 4.x

### Python 依赖
- Python 3.7+
- opencv-python
- numpy

## 编译说明

### Windows
```bash
cd core
build_cpp.bat
```

编译输出：
- `bin/FaceExtractorDLL.dll` - 主 DLL
- `bin/onnxruntime.dll` - ONNX Runtime
- `bin/opencv_world4xx.dll` - OpenCV

## 测试

### 基础测试
```bash
cd tests
python test_phase1_basic.py
```

### 完整验证
```bash
python test_final_verification.py
```

### 对比测试（需要 2DFAN 模型）
```bash
python test_compare_2dfan.py
```

## 性能

### 推理速度（CPU）
- 1k3d68: ~10-15ms（单采样）
- 1k3d68: ~50-75ms（5点多采样）
- 2d106det: ~10-15ms
- 2DFAN: ~15-20ms

### 精度
- 1k3d68: 与 2DFAN 误差待测试
- 2d106det: 稳定可靠

## 文档

- [快速启动指南](tests/QUICK_START.md)
- [实验指南](tests/EXPERIMENT_GUIDE.md)
- [需求文档](.kiro/specs/1k3d68-optimization/requirements.md)
- [设计文档](.kiro/specs/1k3d68-optimization/design.md)
- [任务列表](.kiro/specs/1k3d68-optimization/tasks.md)

## 更新日志

### 2026-02-18
- ✅ 添加模型类型自动检测
- ✅ 实现独立配置管理系统
- ✅ 添加多采样支持
- ✅ 实现输出验证机制
- ✅ 修复裁剪因子配置问题
- ✅ 创建完整测试套件
- ✅ 编写详细文档

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可

[添加许可信息]

## 联系

- GitHub: https://github.com/dayuanzong/cpp_Extract_Revision
- Issues: https://github.com/dayuanzong/cpp_Extract_Revision/issues

---

**最后更新**: 2026年2月18日  
**版本**: 1.0.0  
**状态**: 稳定
