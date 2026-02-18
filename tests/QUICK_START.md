# 1k3d68 优化项目 - 快速启动指南

## 当前状态

✅ **阶段1完成**: 基础重构已完成，所有核心功能已实现并测试通过

## 快速验证

### 1. 验证安装

运行基础测试确认一切正常：

```bash
cd tests
python test_final_verification.py
```

或双击：`tests/test_final_verification.bat`

**预期输出**:
- ✓ 模型类型检测成功
- ✓ 配置管理正常
- ✓ 多采样功能工作
- ✓ 输出验证正常

### 2. 准备实验（可选）

如果要进行精度优化实验，需要：

#### 2.1 准备 2DFAN 模型
将 `2DFAN-4.onnx` 放到：
```
assets/models/FAN/2DFAN-4.onnx
```

#### 2.2 准备测试图片
在 `tests/test_images/` 目录下放置测试图片（.jpg 或 .png）

建议：10-20 张包含清晰人脸的图片

### 3. 运行对比测试（可选）

如果已准备好 2DFAN 模型和测试图片：

```bash
cd tests
python test_compare_2dfan.py
```

或双击：`tests/test_compare_2dfan.bat`

**输出**:
- 误差统计（平均、标准差、最大、最小）
- 可视化对比图（comparison_result.png）
- 详细结果（comparison_result.json）

## 当前配置

### 1k3d68 配置
```cpp
crop_factor: 1.75f       // 裁剪因子（与原来相同，待优化）
norm_mode: MEAN_STD      // 归一化模式（待优化）
multi_sample: enabled    // 多采样（5点）
```

### 2d106det 配置
```cpp
crop_factor: 1.75f       // 裁剪因子（保持不变）
norm_mode: AUTO          // 归一化模式（保持不变）
multi_sample: disabled   // 无多采样
```

## ⚠️ 重要说明

**crop_factor 理解**:
- crop_factor **越大** → 人脸越小（裁剪区域越大）
- crop_factor **越小** → 人脸越大（裁剪区域越小）

当前两个模型都使用 1.75f，这是原始的正确值。

## 下一步

### 选项 A: 直接使用（推荐）

如果当前配置满足需求，可以直接使用：

1. 代码已编译并在 `bin/` 目录
2. Python 接口正常工作
3. 1k3d68 已启用多采样优化

### 选项 B: 进行优化实验

如果需要进一步优化精度：

1. 阅读 `EXPERIMENT_GUIDE.md`
2. 准备 2DFAN 模型和测试图片
3. 运行实验脚本找到最优参数
4. 更新配置并重新编译

## 测试脚本说明

### 基础测试
- `test_phase1_basic.py` - 测试基础功能
- `test_final_verification.py` - 最终验证测试
- `test_validation.py` - 输出验证测试

### 对比测试
- `test_compare_2dfan.py` - 与 2DFAN 对比（需要 2DFAN 模型）

### 实验脚本
- `experiment_crop_factor.py` - 裁剪因子优化
- `experiment_norm_mode.py` - 归一化模式优化
- `experiment_auto.py` - 自动化全面优化

## 常见问题

### Q: 测试显示"No faces detected"
**A**: 这是正常的，如果测试图片不存在或图片中没有人脸。添加测试图片到 `tests/test_images/` 目录。

### Q: 如何知道优化是否有效？
**A**: 运行 `test_compare_2dfan.py` 对比优化前后的误差。目标是平均误差降低 30%。

### Q: 2d106det 会受影响吗？
**A**: 不会。2d106det 使用独立配置，完全不受 1k3d68 优化的影响。

### Q: 多采样会影响性能吗？
**A**: 会有一定影响（约 5x 推理次数），但通过平均提高了稳定性。可以通过配置禁用。

### Q: 如何修改配置？
**A**: 编辑 `core/src/InsightFaceLandmark.cpp` 中的 `InitializeConfigs()` 方法，然后运行 `core/build_cpp.bat` 重新编译。

## 技术支持

### 文档
- `EXPERIMENT_GUIDE.md` - 详细实验指南
- `PHASE1_COMPLETION_REPORT.md` - 阶段1完成报告
- `.kiro/specs/1k3d68-optimization/` - 完整规格文档

### 日志
查看控制台输出，关键信息：
- `Model type detected as 1K3D68` - 模型类型检测
- `Validation failed` - 输出验证失败（如果有）

## 快速命令参考

```bash
# 编译 C++ 代码
cd core
build_cpp.bat

# 运行基础测试
cd tests
python test_final_verification.py

# 运行对比测试（需要 2DFAN）
python test_compare_2dfan.py

# 查看帮助
python test_compare_2dfan.py --help
```

## 项目结构

```
cpp_Extract_Revision/
├── core/
│   ├── src/
│   │   ├── InsightFaceLandmark.h    # 头文件（已修改）
│   │   └── InsightFaceLandmark.cpp  # 实现文件（已修改）
│   └── build_cpp.bat                # 编译脚本
├── bin/                             # 编译输出（DLL）
├── tests/
│   ├── test_final_verification.py   # 快速验证
│   ├── test_compare_2dfan.py        # 对比测试
│   ├── experiment_*.py              # 实验脚本
│   ├── QUICK_START.md               # 本文档
│   ├── EXPERIMENT_GUIDE.md          # 实验指南
│   └── PHASE1_COMPLETION_REPORT.md  # 完成报告
└── .kiro/specs/1k3d68-optimization/ # 规格文档
```

## 成功标志

运行 `test_final_verification.py` 后看到：

```
✓ Phase 1 Complete: Basic Refactoring
  - Model type detection: ✓
  - Configuration management: ✓
  - Extract method refactoring: ✓
  - Multi-sampling implementation: ✓
  - Output validation: ✓

✓ All core features implemented and working!
```

恭喜！系统已准备就绪。

---

**最后更新**: 2026年2月18日  
**版本**: 1.0  
**状态**: 阶段1完成
