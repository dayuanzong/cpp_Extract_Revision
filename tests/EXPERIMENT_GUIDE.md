# 1k3d68 优化实验指南

## 概述

本指南说明如何进行 1k3d68 模型的优化实验，包括裁剪因子和归一化模式的调优。

## 前提条件

1. **2DFAN 模型**: 需要 `assets/models/FAN/2DFAN-4.onnx` 作为参考基准
2. **测试图片**: 在 `tests/test_images/` 目录下放置测试图片（.jpg 或 .png）
3. **Python 环境**: 已安装必要的依赖（opencv-python, numpy）

## 实验脚本

### 1. 快速对比测试 (推荐先运行)

**脚本**: `test_compare_2dfan.py`

**用途**: 快速测试当前配置与 2DFAN 的精度差距

**运行方法**:
```bash
cd tests
python test_compare_2dfan.py
# 或
test_compare_2dfan.bat
```

**输出**:
- 控制台显示误差统计（平均、标准差、最大、最小、中位数）
- `comparison_result.png`: 可视化对比图
- `comparison_result.json`: 详细结果数据

### 2. 裁剪因子实验

**脚本**: `experiment_crop_factor.py`

**用途**: 测试不同裁剪因子（1.4 ~ 2.0）的效果

**手动流程**:
1. 打开 `core/src/InsightFaceLandmark.cpp`
2. 找到 `configs[ModelType::MODEL_1K3D68]` 配置
3. 修改 `crop_factor` 值（例如：1.4f, 1.5f, 1.6f, ...）
4. 运行 `core\build_cpp.bat` 重新编译
5. 运行 `python experiment_crop_factor.py`
6. 记录结果
7. 重复步骤 3-6 测试其他值

**输出**:
- `crop_factor_results.json`: 所有测试结果
- 控制台显示最佳配置

### 3. 归一化模式实验

**脚本**: `experiment_norm_mode.py`

**用途**: 测试不同归一化模式（ZERO_ONE vs MEAN_STD）的效果

**手动流程**:
1. 打开 `core/src/InsightFaceLandmark.cpp`
2. 找到 `configs[ModelType::MODEL_1K3D68]` 配置
3. 修改 `norm_mode` 值：
   - `NormMode::ZERO_ONE` - [0,1] 归一化
   - `NormMode::MEAN_STD` - (x-127.5)/128 归一化
4. 运行 `core\build_cpp.bat` 重新编译
5. 运行 `python experiment_norm_mode.py`
6. 输入当前配置的模式编号
7. 记录结果
8. 重复步骤 3-7 测试另一种模式

**输出**:
- `norm_mode_results.json`: 所有测试结果
- 控制台显示最佳配置

### 4. 自动化实验 (高级)

**脚本**: `experiment_auto.py`

**用途**: 自动测试所有配置组合（裁剪因子 × 归一化模式）

**运行方法**:
```bash
cd tests
python experiment_auto.py
```

**注意**:
- 此脚本会自动修改 C++ 代码并重新编译
- 会备份原始配置并在完成后恢复
- 测试时间较长（每个配置需要编译+测试）
- 建议在测试前备份代码

**输出**:
- `optimization_results.json`: 所有配置的测试结果
- 控制台显示最佳配置

## 当前配置

根据 `core/src/InsightFaceLandmark.cpp` 中的初始配置：

```cpp
configs[ModelType::MODEL_1K3D68] = {
    ModelType::MODEL_1K3D68,
    {
        1.75f,  // crop_factor - same as original (NOT 1.6f!)
        true,  // use_padding
        cv::Scalar(127, 127, 127)  // pad_value
    },
    {
        NormMode::MEAN_STD,  // norm_mode - to be optimized
        127.5f,  // mean
        128.0f   // std
    },
    {
        true,   // enabled - multi-sampling enabled
        5,      // sample_count - 5-point sampling
        1.0f    // offset_pixels
    }
};
```

**重要**: crop_factor 当前设置为 1.75f（与原来相同）。这是正确的起点。

**crop_factor 理解**:
- 公式: `s = input_w / (size * crop_factor)`
- crop_factor **越大** → s 越小 → 裁剪区域越大 → 人脸越小
- crop_factor **越小** → s 越大 → 裁剪区域越小 → 人脸越大

**优化方向**:
- 如果人脸太小，需要**减小** crop_factor (如 1.6f, 1.5f)
- 如果人脸太大，需要**增大** crop_factor (如 1.8f, 1.9f)

## 评估指标

### 主要指标
- **Mean Error**: 平均欧氏距离误差（像素）- 越小越好
- **Std Error**: 标准差 - 越小越稳定
- **Max Error**: 最大误差 - 反映最坏情况

### 次要指标
- **Inference Time**: 推理时间（毫秒）- 不应超过 50% 增长
- **Median Error**: 中位数误差 - 反映典型情况

## 优化目标

根据需求文档：

1. **精度提升**: 平均误差降低至少 30%
2. **接近 2DFAN**: 与 2DFAN 的误差差距缩小至 50% 以内
3. **性能可接受**: 推理时间增加不超过 50%
4. **稳定性提升**: 标准差降低至少 20%

## 实验建议

### 阶段 1: 基线测试
1. 运行 `test_compare_2dfan.py` 获取当前配置的基线
2. 记录当前的误差和推理时间

### 阶段 2: 裁剪因子优化
1. 测试范围：1.4 ~ 2.0，步长 0.1
2. 固定归一化模式为 MEAN_STD
3. 找到误差最小的裁剪因子

### 阶段 3: 归一化模式优化
1. 使用最优裁剪因子
2. 测试 ZERO_ONE 和 MEAN_STD
3. 选择误差更小的模式

### 阶段 4: 联合优化（可选）
1. 如果单独优化效果不理想
2. 运行 `experiment_auto.py` 测试所有组合
3. 找到全局最优配置

### 阶段 5: 验证
1. 使用最优配置重新编译
2. 在更大的测试集上验证
3. 确认达到优化目标

## 结果记录

建议创建实验记录表格：

| 配置 | Crop Factor | Norm Mode | Mean Error | Std Error | Max Error | Time (ms) | 备注 |
|------|-------------|-----------|------------|-----------|-----------|-----------|------|
| 基线 | 1.6 | MEAN_STD | ? | ? | ? | ? | 当前配置 |
| 实验1 | 1.4 | MEAN_STD | ? | ? | ? | ? | |
| 实验2 | 1.5 | MEAN_STD | ? | ? | ? | ? | |
| ... | ... | ... | ... | ... | ... | ... | |

## 故障排除

### 问题：2DFAN 模型未找到
**解决**: 确保 `assets/models/FAN/2DFAN-4.onnx` 存在

### 问题：测试图片未找到
**解决**: 在 `tests/test_images/` 目录下添加测试图片

### 问题：编译失败
**解决**: 
1. 检查 C++ 代码语法
2. 确保 Visual Studio 和 CMake 正确安装
3. 查看编译错误信息

### 问题：DLL 加载失败
**解决**:
1. 确保编译成功
2. 检查 `bin/` 目录下的 DLL 文件
3. 重新运行 `core\build_cpp.bat`

## 下一步

完成实验后：
1. 更新 `InsightFaceLandmark.cpp` 中的最优配置
2. 重新编译并测试
3. 更新文档记录最优参数
4. 进入阶段 3：验证和测试
