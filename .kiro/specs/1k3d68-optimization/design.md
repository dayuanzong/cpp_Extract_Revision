# 1k3d68 模型精度优化设计文档

## 1. 设计概述

本设计文档详细说明如何优化 1k3d68.onnx 模型的精度，使其接近 2DFAN.onnx 的效果。优化策略包括：裁剪参数独立化、预处理模式锁定、多采样平均和输出一致性校验。

### 1.1 设计目标

- **精度提升**: 平均误差降低至少 30%
- **接近 2DFAN**: 与 2DFAN 的误差差距缩小至 50% 以内
- **性能可接受**: 推理时间增加不超过 50%
- **向后兼容**: 不影响 2d106det 的现有功能

### 1.2 技术栈

- **语言**: C++17
- **框架**: ONNX Runtime
- **图像处理**: OpenCV 4.x
- **构建系统**: CMake

## 2. 架构设计

### 2.1 当前架构分析

```
InsightFaceLandmark
├── Constructor: 初始化 ONNX Session
├── Extract: 主入口，执行推理
│   ├── Crop: 裁剪人脸区域
│   ├── 预处理: 归一化（自动选择模式）
│   ├── 推理: ONNX Runtime 执行
│   └── PostProcess: 坐标转换
└── 共享参数: 1k3d68 和 2d106det 使用相同裁剪比例
```

**问题点**:
1. 裁剪比例 1.75f 是折中值，不是 1k3d68 的最优值
2. 归一化模式自动选择，可能不稳定
3. 无多采样，精度受单次裁剪影响
4. 无输出校验，异常值可能影响结果

### 2.2 优化后架构

```
InsightFaceLandmark
├── Constructor
│   ├── 初始化 ONNX Session
│   └── 检测模型类型（1k3d68 / 2d106det）
├── Extract (优化)
│   ├── 多采样控制（可选）
│   ├── ExtractSingle (新增)
│   │   ├── Crop (优化)
│   │   │   └── 模型特定裁剪比例
│   │   ├── 预处理 (优化)
│   │   │   └── 模型特定归一化模式
│   │   ├── 推理
│   │   └── PostProcess
│   ├── 多采样平均 (新增)
│   └── 输出校验 (新增)
└── 配置管理
    ├── 1k3d68 专用配置
    └── 2d106det 专用配置
```

## 3. 详细设计

### 3.1 模型类型检测

**目的**: 自动识别模型类型，应用对应的优化策略。

**实现方案**:

```cpp
enum class ModelType {
    UNKNOWN,
    MODEL_1K3D68,    // 68点，3D坐标
    MODEL_2D106DET   // 106点，2D坐标
};

class InsightFaceLandmark {
private:
    ModelType model_type = ModelType::UNKNOWN;
    
    // 在构造函数中检测
    void DetectModelType() {
        // 方法1: 通过模型文件名检测
        // 方法2: 通过输出形状检测
        // 方法3: 通过第一次推理结果检测
    }
};
```

**检测逻辑**:
1. **文件名检测**: 如果路径包含 "1k3d68" → MODEL_1K3D68
2. **输出形状检测**: 
   - 输出 68×3 或 204 → MODEL_1K3D68
   - 输出 106×2 或 212 → MODEL_2D106DET
3. **延迟检测**: 第一次推理后根据输出确定

### 3.2 裁剪比例独立化

**当前问题**: 
- 1k3d68 和 2d106det 共用 1.75f 裁剪因子
- 测试显示 1k3d68 需要更小的裁剪因子（更大的裁剪区域）

**优化方案**:

```cpp
struct CropConfig {
    float crop_factor;      // 裁剪因子
    bool use_padding;       // 是否使用边界填充
    cv::Scalar pad_value;   // 填充值
};

class InsightFaceLandmark {
private:
    CropConfig crop_config_1k3d68 = {
        .crop_factor = 1.6f,    // 待实验确定最优值
        .use_padding = true,
        .pad_value = cv::Scalar(127, 127, 127)
    };
    
    CropConfig crop_config_2d106det = {
        .crop_factor = 1.75f,   // 保持现有值
        .use_padding = true,
        .pad_value = cv::Scalar(127, 127, 127)
    };
    
    const CropConfig& GetCropConfig() const {
        switch (model_type) {
            case ModelType::MODEL_1K3D68:
                return crop_config_1k3d68;
            case ModelType::MODEL_2D106DET:
                return crop_config_2d106det;
            default:
                return crop_config_2d106det; // 默认
        }
    }
};
```

**裁剪因子实验计划**:
- 测试范围: 1.4f ~ 2.0f，步长 0.1f
- 评估指标: 与 2DFAN 的平均欧氏距离
- 目标: 找到使误差最小的因子

### 3.3 预处理模式锁定

**当前问题**:
- 自动尝试两种归一化模式（0..1 和 (x-127.5)/128）
- 选择得分高的结果，可能不稳定

**优化方案**:

```cpp
enum class NormMode {
    AUTO,           // 自动选择（当前行为）
    ZERO_ONE,       // [0, 1] 归一化
    MEAN_STD        // (x-127.5)/128 归一化
};

struct PreprocessConfig {
    NormMode norm_mode;
    float mean;
    float std;
};

class InsightFaceLandmark {
private:
    PreprocessConfig preprocess_config_1k3d68 = {
        .norm_mode = NormMode::MEAN_STD,  // 待实验确定
        .mean = 127.5f,
        .std = 128.0f
    };
    
    PreprocessConfig preprocess_config_2d106det = {
        .norm_mode = NormMode::AUTO,  // 保持现有行为
        .mean = 0.0f,
        .std = 1.0f
    };
    
    const PreprocessConfig& GetPreprocessConfig() const {
        switch (model_type) {
            case ModelType::MODEL_1K3D68:
                return preprocess_config_1k3d68;
            case ModelType::MODEL_2D106DET:
                return preprocess_config_2d106det;
            default:
                return preprocess_config_2d106det;
        }
    }
};
```

**归一化模式实验**:
1. 测试 ZERO_ONE 模式的精度
2. 测试 MEAN_STD 模式的精度
3. 选择精度更高的模式锁定

### 3.4 多采样平均

**参考**: 2DFAN 使用 5 点采样（中心 + 上下左右微调）

**实现方案**:

```cpp
struct MultiSampleConfig {
    bool enabled;           // 是否启用
    int sample_count;       // 采样点数（1, 5, 9）
    float offset_pixels;    // 偏移像素数
};

class InsightFaceLandmark {
private:
    MultiSampleConfig multi_sample_config_1k3d68 = {
        .enabled = true,
        .sample_count = 5,
        .offset_pixels = 1.0f
    };
    
    MultiSampleConfig multi_sample_config_2d106det = {
        .enabled = false,  // 保持现有行为
        .sample_count = 1,
        .offset_pixels = 0.0f
    };
    
    // 单次推理（提取为独立方法）
    std::vector<cv::Point2f> ExtractSingle(
        const cv::Mat& img, 
        const cv::Rect2f& face_rect,
        const cv::Point2f& center_offset = cv::Point2f(0, 0)
    );
    
    // 多采样主方法
    std::vector<cv::Point2f> ExtractWithMultiSample(
        const cv::Mat& img,
        const cv::Rect2f& face_rect
    ) {
        const auto& config = GetMultiSampleConfig();
        if (!config.enabled || config.sample_count <= 1) {
            return ExtractSingle(img, face_rect);
        }
        
        // 生成采样点
        std::vector<cv::Point2f> offsets;
        offsets.push_back(cv::Point2f(0, 0));  // 中心
        
        if (config.sample_count >= 5) {
            float off = config.offset_pixels;
            offsets.push_back(cv::Point2f(-off, 0));   // 左
            offsets.push_back(cv::Point2f(off, 0));    // 右
            offsets.push_back(cv::Point2f(0, -off));   // 上
            offsets.push_back(cv::Point2f(0, off));    // 下
        }
        
        if (config.sample_count >= 9) {
            float off = config.offset_pixels;
            offsets.push_back(cv::Point2f(-off, -off)); // 左上
            offsets.push_back(cv::Point2f(off, -off));  // 右上
            offsets.push_back(cv::Point2f(-off, off));  // 左下
            offsets.push_back(cv::Point2f(off, off));   // 右下
        }
        
        // 多次采样并平均
        std::vector<cv::Point2f> accum;
        int valid_count = 0;
        
        for (const auto& offset : offsets) {
            auto pts = ExtractSingle(img, face_rect, offset);
            if (pts.empty()) continue;
            
            if (accum.empty()) {
                accum = pts;
            } else {
                for (size_t i = 0; i < pts.size() && i < accum.size(); i++) {
                    accum[i].x += pts[i].x;
                    accum[i].y += pts[i].y;
                }
            }
            valid_count++;
        }
        
        // 求平均
        if (valid_count > 0) {
            for (auto& pt : accum) {
                pt.x /= (float)valid_count;
                pt.y /= (float)valid_count;
            }
        }
        
        return accum;
    }
};
```

**采样策略**:
- **5点采样**: 中心 + 上下左右各偏移 1 像素
- **9点采样**: 5点 + 四个对角线方向
- **偏移量**: 可配置，默认 1.0 像素

### 3.5 输出一致性校验

**目的**: 检测和处理异常输出。

**实现方案**:

```cpp
struct ValidationResult {
    bool valid;
    std::string error_message;
    std::vector<int> invalid_indices;  // 异常点索引
};

class InsightFaceLandmark {
private:
    ValidationResult ValidateOutput(
        const std::vector<cv::Point2f>& landmarks,
        const cv::Rect2f& face_rect,
        const cv::Size& img_size
    ) {
        ValidationResult result{true, "", {}};
        
        // 1. 检查点数
        if (landmarks.size() != 68) {
            result.valid = false;
            result.error_message = "Invalid landmark count: " + 
                                  std::to_string(landmarks.size());
            return result;
        }
        
        // 2. 检查坐标范围（扩展人脸框）
        float x1 = face_rect.x - face_rect.width * 0.5f;
        float y1 = face_rect.y - face_rect.height * 0.5f;
        float x2 = face_rect.x + face_rect.width * 2.5f;
        float y2 = face_rect.y + face_rect.height * 2.5f;
        
        for (size_t i = 0; i < landmarks.size(); i++) {
            const auto& pt = landmarks[i];
            
            // 检查 NaN/Inf
            if (std::isnan(pt.x) || std::isnan(pt.y) ||
                std::isinf(pt.x) || std::isinf(pt.y)) {
                result.valid = false;
                result.invalid_indices.push_back((int)i);
                continue;
            }
            
            // 检查是否在合理范围内
            if (pt.x < x1 || pt.x > x2 || pt.y < y1 || pt.y > y2) {
                result.invalid_indices.push_back((int)i);
            }
        }
        
        // 3. 检查异常点比例
        float invalid_ratio = (float)result.invalid_indices.size() / 
                             (float)landmarks.size();
        if (invalid_ratio > 0.3f) {  // 超过30%异常
            result.valid = false;
            result.error_message = "Too many invalid points: " + 
                                  std::to_string(result.invalid_indices.size());
        }
        
        return result;
    }
    
    // 在 Extract 中使用
    std::vector<cv::Point2f> Extract(
        const cv::Mat& img, 
        const cv::Rect2f& face_rect
    ) {
        auto landmarks = ExtractWithMultiSample(img, face_rect);
        
        // 校验输出
        auto validation = ValidateOutput(landmarks, face_rect, img.size());
        if (!validation.valid) {
            std::cerr << "Landmark validation failed: " 
                     << validation.error_message << std::endl;
            // 可选：返回空结果或使用回退策略
        }
        
        return landmarks;
    }
};
```

### 3.6 配置管理

**目的**: 集中管理所有配置参数，便于调整和实验。

**实现方案**:

```cpp
struct ModelConfig {
    ModelType type;
    CropConfig crop;
    PreprocessConfig preprocess;
    MultiSampleConfig multi_sample;
    
    // 从 JSON 或配置文件加载（可选）
    static ModelConfig LoadFromFile(const std::string& path);
    void SaveToFile(const std::string& path) const;
};

class InsightFaceLandmark {
private:
    std::map<ModelType, ModelConfig> configs;
    
    void InitializeConfigs() {
        // 1k3d68 配置
        configs[ModelType::MODEL_1K3D68] = {
            .type = ModelType::MODEL_1K3D68,
            .crop = {
                .crop_factor = 1.6f,
                .use_padding = true,
                .pad_value = cv::Scalar(127, 127, 127)
            },
            .preprocess = {
                .norm_mode = NormMode::MEAN_STD,
                .mean = 127.5f,
                .std = 128.0f
            },
            .multi_sample = {
                .enabled = true,
                .sample_count = 5,
                .offset_pixels = 1.0f
            }
        };
        
        // 2d106det 配置（保持现有行为）
        configs[ModelType::MODEL_2D106DET] = {
            .type = ModelType::MODEL_2D106DET,
            .crop = {
                .crop_factor = 1.75f,
                .use_padding = true,
                .pad_value = cv::Scalar(127, 127, 127)
            },
            .preprocess = {
                .norm_mode = NormMode::AUTO,
                .mean = 0.0f,
                .std = 1.0f
            },
            .multi_sample = {
                .enabled = false,
                .sample_count = 1,
                .offset_pixels = 0.0f
            }
        };
    }
    
    const ModelConfig& GetConfig() const {
        auto it = configs.find(model_type);
        if (it != configs.end()) {
            return it->second;
        }
        return configs.at(ModelType::MODEL_2D106DET);  // 默认
    }
};
```

## 4. 验证与测试

### 4.1 对比测试脚本

**文件**: `tests/test_1k3d68_optimization.py`

**功能**:
1. 加载测试图片集
2. 分别使用 1k3d68（优化前/后）和 2DFAN 提取关键点
3. 计算误差指标
4. 生成对比报告

**指标**:
```python
def calculate_metrics(pred_landmarks, gt_landmarks):
    """
    计算精度指标
    
    Args:
        pred_landmarks: 预测的关键点 (N, 68, 2)
        gt_landmarks: 参考关键点 (N, 68, 2)
    
    Returns:
        dict: 包含各种指标
    """
    # 欧氏距离
    distances = np.linalg.norm(pred_landmarks - gt_landmarks, axis=2)
    
    return {
        'mean_error': np.mean(distances),
        'std_error': np.std(distances),
        'max_error': np.max(distances),
        'median_error': np.median(distances),
        'per_point_error': np.mean(distances, axis=0)  # 每个点的平均误差
    }
```

### 4.2 可视化工具

**文件**: `tests/visualize_landmarks.py`

**功能**:
1. 在图片上绘制不同模型的关键点
2. 使用不同颜色区分
3. 标注误差较大的点
4. 生成对比图

**示例**:
```python
def visualize_comparison(img, landmarks_dict):
    """
    可视化多个模型的关键点对比
    
    Args:
        img: 原始图片
        landmarks_dict: {
            '2DFAN': landmarks,
            '1k3d68_before': landmarks,
            '1k3d68_after': landmarks
        }
    """
    colors = {
        '2DFAN': (0, 255, 0),        # 绿色（参考）
        '1k3d68_before': (255, 0, 0), # 蓝色（优化前）
        '1k3d68_after': (0, 0, 255)   # 红色（优化后）
    }
    
    # 绘制关键点和连线
    # 计算并标注误差
    # 保存对比图
```

### 4.3 自动化测试

**文件**: `tests/test_1k3d68_regression.cpp`

**功能**:
1. 单元测试：测试各个优化组件
2. 集成测试：测试完整流程
3. 回归测试：确保不影响 2d106det

**测试用例**:
```cpp
TEST(InsightFaceLandmark, ModelTypeDetection) {
    // 测试模型类型检测
}

TEST(InsightFaceLandmark, CropFactorIndependence) {
    // 测试裁剪因子独立性
}

TEST(InsightFaceLandmark, MultiSampling) {
    // 测试多采样功能
}

TEST(InsightFaceLandmark, OutputValidation) {
    // 测试输出校验
}

TEST(InsightFaceLandmark, BackwardCompatibility) {
    // 测试向后兼容性（2d106det）
}
```

## 5. 性能优化

### 5.1 多采样性能优化

**问题**: 5点采样会导致推理次数增加 5 倍

**优化策略**:
1. **批处理**: 如果 ONNX Runtime 支持，将 5 次推理合并为批处理
2. **早停**: 如果前几次采样结果一致性高，提前停止
3. **自适应采样**: 根据人脸大小和质量决定采样点数

```cpp
std::vector<cv::Point2f> ExtractWithAdaptiveSampling(
    const cv::Mat& img,
    const cv::Rect2f& face_rect
) {
    // 1. 先进行单次采样
    auto center_result = ExtractSingle(img, face_rect);
    
    // 2. 评估结果质量
    float quality = EvaluateQuality(center_result, face_rect);
    
    // 3. 如果质量高，直接返回
    if (quality > 0.9f) {
        return center_result;
    }
    
    // 4. 否则进行多采样
    return ExtractWithMultiSample(img, face_rect);
}
```

### 5.2 内存优化

**策略**:
1. 复用输入 tensor 内存
2. 避免不必要的内存拷贝
3. 使用对象池管理临时对象

## 6. 实施计划

### 阶段 1: 基础重构（1-2天）
- [ ] 添加模型类型检测
- [ ] 重构 Extract 方法，提取 ExtractSingle
- [ ] 添加配置管理结构

### 阶段 2: 裁剪和预处理优化（2-3天）
- [ ] 实现裁剪比例独立化
- [ ] 实验确定最优裁剪因子
- [ ] 实现预处理模式锁定
- [ ] 实验确定最优归一化模式

### 阶段 3: 多采样实现（2-3天）
- [ ] 实现基础多采样功能
- [ ] 实现自适应采样
- [ ] 性能优化

### 阶段 4: 验证和测试（2-3天）
- [ ] 实现输出校验
- [ ] 创建对比测试脚本
- [ ] 创建可视化工具
- [ ] 编写单元测试

### 阶段 5: 调优和文档（1-2天）
- [ ] 参数调优
- [ ] 性能测试
- [ ] 编写文档
- [ ] 代码审查

**总计**: 8-13 天

## 7. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 优化后精度未达预期 | 多轮实验，调整参数；保留回退机制 |
| 性能损失过大 | 实现自适应采样；提供开关控制 |
| 影响 2d106det | 充分的回归测试；代码隔离 |
| 参数调优困难 | 自动化调参脚本；网格搜索 |

## 8. 可测试性设计

### 8.1 单元测试覆盖

- 模型类型检测
- 裁剪函数（不同配置）
- 预处理函数（不同模式）
- 多采样逻辑
- 输出校验逻辑

### 8.2 集成测试

- 完整推理流程
- 多模型对比
- 性能基准测试

### 8.3 回归测试

- 2d106det 功能不变
- 接口兼容性
- 性能不退化

## 9. 文档和注释

### 9.1 代码注释

- 每个配置参数的含义和取值范围
- 关键算法的原理说明
- 实验结果和参数选择依据

### 9.2 用户文档

- 如何使用优化后的模型
- 配置参数说明
- 性能和精度对比

### 9.3 开发文档

- 架构设计说明
- 实验记录和结果
- 未来优化方向

## 10. 正确性属性

### 属性 1: 模型类型检测正确性
**描述**: 模型类型检测必须准确识别 1k3d68 和 2d106det

**形式化**:
```
∀ model_path: 
  if contains(model_path, "1k3d68") then DetectModelType() = MODEL_1K3D68
  if contains(model_path, "2d106det") then DetectModelType() = MODEL_2D106DET
```

**验证**: Requirements 1.1

### 属性 2: 配置独立性
**描述**: 1k3d68 和 2d106det 的配置必须独立，互不影响

**形式化**:
```
∀ config_1k3d68, config_2d106det:
  Modify(config_1k3d68) ⇒ ¬Affects(config_2d106det)
```

**验证**: Requirements 1.1, 2.1

### 属性 3: 多采样一致性
**描述**: 多采样结果必须是各采样点的平均值

**形式化**:
```
∀ samples: [s1, s2, ..., sn]:
  MultiSample(samples) = Average(s1, s2, ..., sn)
```

**验证**: Requirements 3.1

### 属性 4: 输出有效性
**描述**: 输出关键点必须在合理范围内

**形式化**:
```
∀ landmark ∈ landmarks:
  ¬(isnan(landmark.x) ∨ isnan(landmark.y) ∨ 
    isinf(landmark.x) ∨ isinf(landmark.y))
  ∧ InBounds(landmark, face_rect, margin)
```

**验证**: Requirements 4.1

### 属性 5: 向后兼容性
**描述**: 2d106det 的行为必须保持不变

**形式化**:
```
∀ img, face_rect:
  Extract_2d106det_before(img, face_rect) = 
  Extract_2d106det_after(img, face_rect)
```

**验证**: Requirements 1.1, NFR2

## 11. 附录

### 11.1 参考代码位置

- InsightFaceLandmark: `core/src/InsightFaceLandmark.{h,cpp}`
- FANExtractor (参考): `core/src/FANExtractor.{h,cpp}`
- 测试脚本: `tests/test_face_proportion.py`

### 11.2 实验数据格式

```json
{
  "experiment_id": "crop_factor_test_001",
  "model": "1k3d68",
  "config": {
    "crop_factor": 1.6,
    "norm_mode": "MEAN_STD",
    "multi_sample": true
  },
  "results": {
    "mean_error": 2.34,
    "std_error": 1.12,
    "max_error": 8.45,
    "inference_time_ms": 15.6
  }
}
```

### 11.3 性能基准

| 配置 | 推理时间 (ms) | 平均误差 (px) |
|------|--------------|--------------|
| 当前 1k3d68 | 10.2 | 5.8 |
| 2DFAN | 18.5 | 2.1 |
| 目标 1k3d68 | < 15.3 | < 4.0 |
