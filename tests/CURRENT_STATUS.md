# 当前状态报告

## 问题
用户报告：1k3d68 提取的图片只包含一个五官（人脸被裁剪得太小）

## 已采取的措施

### 第一次修复尝试
- 将 crop_factor 从 1.6f 改回 1.75f
- 结果：**问题依然存在**

### 第二次修复尝试（当前）
- 将所有配置恢复到原始状态
- 禁用多采样
- 将归一化模式改回 AUTO
- 结果：**等待测试**

## 当前配置

### 1k3d68
```cpp
configs[ModelType::MODEL_1K3D68] = {
    ModelType::MODEL_1K3D68,
    {
        1.75f,  // crop_factor - 与原来完全相同
        true,   // use_padding
        cv::Scalar(127, 127, 127)
    },
    {
        NormMode::AUTO,  // norm_mode - 与原来完全相同
        0.0f,   // mean
        1.0f    // std
    },
    {
        false,  // enabled - 禁用多采样
        1,      // sample_count
        0.0f    // offset_pixels
    }
};
```

### 2d106det
```cpp
configs[ModelType::MODEL_2D106DET] = {
    ModelType::MODEL_2D106DET,
    {
        1.75f,  // crop_factor
        true,   // use_padding
        cv::Scalar(127, 127, 127)
    },
    {
        NormMode::AUTO,  // norm_mode
        0.0f,   // mean
        1.0f    // std
    },
    {
        false,  // enabled
        1,      // sample_count
        0.0f    // offset_pixels
    }
};
```

## 代码修改对比

### 原始代码（推测）
```cpp
cv::Mat Crop(const cv::Mat& img, const cv::Rect2f& rect, cv::Mat& M_inv) {
    cv::Point2f center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
    float size = std::max(rect.width, rect.height);
    float s = (float)input_w / (size * 1.75f);  // 硬编码 1.75f
    ...
}
```

### 当前代码
```cpp
cv::Mat Crop(const cv::Mat& img, const cv::Rect2f& rect, cv::Mat& M_inv, const cv::Point2f& center_offset) {
    cv::Point2f center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
    center.x += center_offset.x;  // 新增：支持偏移
    center.y += center_offset.y;
    
    float size = std::max(rect.width, rect.height);
    const auto& config = GetConfig();  // 新增：从配置获取
    float crop_factor = config.crop.crop_factor;  // 应该是 1.75f
    float s = (float)input_w / (size * crop_factor);
    ...
}
```

## 可能的问题

### 1. center_offset 影响
即使 center_offset 是 (0, 0)，添加这个逻辑可能有副作用

### 2. GetConfig() 调用
配置可能在某些情况下未正确初始化或返回错误值

### 3. 模型类型检测错误
如果模型类型检测错误，可能使用了错误的配置

### 4. 代码重构引入的 bug
重构过程中可能引入了其他 bug

## 诊断步骤

### 步骤 1: 检查模型类型检测
查看控制台输出：
```
InsightFaceLandmark: Model type detected as 1K3D68
```

如果显示 "2D106DET" 或 "UNKNOWN"，说明检测有问题

### 步骤 2: 检查 crop_factor 值
在 Crop 方法中添加调试输出：
```cpp
std::cout << "DEBUG: crop_factor = " << crop_factor << std::endl;
```

应该输出 1.75

### 步骤 3: 检查裁剪区域
在 Crop 方法中添加调试输出：
```cpp
std::cout << "DEBUG: size = " << size << ", s = " << s << std::endl;
```

### 步骤 4: 对比原始代码
如果有原始代码备份，对比 Crop 方法的实现

## 下一步行动

### 如果当前修复有效
- 逐步启用功能（先启用模型检测，再启用多采样等）
- 找出导致问题的具体功能

### 如果当前修复无效
- 完全回滚所有修改
- 参考 `ROLLBACK_INSTRUCTIONS.md`
- 重新评估优化方案

## 测试方法

### 方法 1: 使用您的应用程序
直接测试人脸提取功能，检查输出目录：
```
D:\Program Files\face_classification\TensorFlow_Extract_Revision\data\output\cpp版本\aligned
```

### 方法 2: 使用测试脚本
```bash
cd tests
python test_crop_fix.py
```

### 方法 3: 视觉检查
检查提取的人脸图片：
- ✅ 应该包含：额头、下巴、耳朵、完整五官
- ✗ 不应该：只有眼睛、鼻子、嘴巴

## 编译状态
✅ 已重新编译成功  
✅ DLL 已更新到 bin/ 目录  
⏳ 等待测试反馈

---

**更新时间**: 2026年2月18日  
**状态**: 等待用户测试反馈  
**配置**: 已恢复到最保守状态
