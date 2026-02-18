# 回滚说明

## 当前状态

已将所有修改回滚到最保守的配置：

### 1k3d68 配置
```cpp
crop_factor: 1.75f       // 与原来完全相同
norm_mode: AUTO          // 与原来完全相同
multi_sample: disabled   // 禁用（原来没有）
```

### 2d106det 配置
```cpp
crop_factor: 1.75f       // 未改变
norm_mode: AUTO          // 未改变
multi_sample: disabled   // 未改变
```

## 如果问题依然存在

这意味着代码重构本身有问题。需要完全回滚所有修改。

### 完全回滚步骤

1. **备份当前文件**（如果还没有）
   ```bash
   copy core\src\InsightFaceLandmark.h core\src\InsightFaceLandmark.h.new
   copy core\src\InsightFaceLandmark.cpp core\src\InsightFaceLandmark.cpp.new
   ```

2. **恢复原始文件**
   
   如果有 git：
   ```bash
   git checkout core/src/InsightFaceLandmark.h
   git checkout core/src/InsightFaceLandmark.cpp
   ```
   
   如果没有 git，需要手动恢复或从备份恢复

3. **重新编译**
   ```bash
   cd core
   build_cpp.bat
   ```

4. **测试验证**
   使用您的应用程序测试人脸提取功能

## 问题诊断

如果回滚后问题解决，说明是我的修改导致的。可能的原因：

### 1. Crop 方法修改
我添加了 `center_offset` 参数，可能影响了裁剪逻辑

### 2. GetConfig() 调用
在 Crop 方法中调用 `GetConfig()` 可能有问题

### 3. 配置初始化时机
配置可能在某些情况下未正确初始化

## 最小化修改方案

如果需要保留某些功能，可以尝试最小化修改：

### 方案 A: 只保留模型类型检测
- 保留 ModelType 枚举
- 保留 DetectModelType() 方法
- 移除所有配置管理
- 移除多采样
- 移除输出验证

### 方案 B: 完全回滚
- 删除所有新增代码
- 恢复到原始状态
- 放弃优化计划

## 联系支持

如果需要进一步帮助，请提供：

1. 提取的人脸图片示例
2. 原始图片
3. 使用的模型（1k3d68 还是 2d106det）
4. 控制台输出日志

## 当前编译状态

✅ 已编译成功
✅ DLL 已更新到 bin/ 目录

## 测试命令

```bash
# 使用您的应用程序测试
# 或
cd tests
python test_crop_fix.py
```

---

**更新时间**: 2026年2月18日  
**状态**: 已回滚到最保守配置  
**下一步**: 等待用户测试反馈
