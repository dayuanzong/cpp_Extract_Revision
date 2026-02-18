# Git 提交指南

## 快速提交步骤

### 方法 1: 使用自动脚本（推荐）

1. **双击运行**:
   ```
   git_setup_and_push.bat
   ```

2. **按照提示操作**:
   - 脚本会自动初始化 git
   - 添加所有文件
   - 提示输入提交信息（可以直接回车使用默认）
   - 推送到 GitHub

3. **如果推送失败**:
   - 确保已在浏览器登录 GitHub
   - 确保仓库 `dayuanzong/cpp_Extract_Revision` 已创建
   - 重新运行脚本

### 方法 2: 手动操作

如果自动脚本有问题，可以手动执行：

```bash
# 1. 初始化 git（如果还没有）
git init

# 2. 配置用户信息
git config user.name "dayuanzong"
git config user.email "your-email@example.com"

# 3. 添加远程仓库
git remote add origin https://github.com/dayuanzong/cpp_Extract_Revision.git

# 4. 添加所有文件
git add .

# 5. 查看将要提交的文件
git status

# 6. 提交
git commit -m "feat: Add 1k3d68 optimization framework with model type detection and configuration management"

# 7. 推送到 GitHub
git branch -M main
git push -u origin main
```

## 提交信息建议

本次提交的主要内容：

```
feat: Add 1k3d68 optimization framework with model type detection and configuration management

- Add automatic model type detection (1k3d68 / 2d106det)
- Implement independent configuration management system
- Add multi-sampling support (configurable)
- Implement output validation mechanism
- Add comprehensive test suite
- Fix crop factor configuration issue
- Create detailed documentation

Features:
- Model type auto-detection
- Independent config for each model
- Multi-sampling (5-point, 9-point)
- Output validation
- Backward compatible with 2d106det

Status: Stable and tested
```

## 如果遇到问题

### 问题 1: 仓库不存在

**解决方法**:
1. 访问 https://github.com/new
2. 创建新仓库 `cpp_Extract_Revision`
3. 不要初始化 README、.gitignore 或 license
4. 重新运行推送脚本

### 问题 2: 认证失败

**解决方法**:
1. 确保在浏览器中登录 GitHub
2. 使用 GitHub Desktop 或 Git Credential Manager
3. 或使用 Personal Access Token

### 问题 3: 文件太大

**解决方法**:
1. 检查 `.gitignore` 是否正确
2. 确保 `bin/` 和 `data/` 目录被忽略
3. 如果模型文件太大，考虑使用 Git LFS

### 问题 4: 推送被拒绝

**解决方法**:
```bash
# 如果远程有内容，先拉取
git pull origin main --allow-unrelated-histories

# 解决冲突后再推送
git push -u origin main
```

## 后续更新

以后更新代码时：

```bash
# 1. 查看修改
git status

# 2. 添加修改的文件
git add .

# 3. 提交
git commit -m "描述你的修改"

# 4. 推送
git push
```

## 查看提交历史

```bash
# 查看提交日志
git log --oneline

# 查看详细日志
git log

# 查看某个文件的历史
git log -- core/src/InsightFaceLandmark.cpp
```

## 分支管理

```bash
# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout main

# 合并分支
git merge feature/new-feature

# 删除分支
git branch -d feature/new-feature
```

## 有用的命令

```bash
# 查看远程仓库
git remote -v

# 查看当前分支
git branch

# 撤销未提交的修改
git checkout -- filename

# 查看差异
git diff

# 查看某次提交的内容
git show commit-hash
```

## 需要帮助？

- Git 官方文档: https://git-scm.com/doc
- GitHub 帮助: https://docs.github.com/
- Git 教程: https://www.atlassian.com/git/tutorials

---

**提示**: 第一次推送可能需要几分钟，请耐心等待。
