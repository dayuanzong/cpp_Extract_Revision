"""
测试抽屉式侧边栏 UI
"""
import sys
from pathlib import Path

# 添加路径
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir / "sdk"))

# 设置环境
import os
os.environ["PATH"] = str(root_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

try:
    os.add_dll_directory(str(root_dir / "bin"))
except Exception:
    pass

# 导入 UI
from core_logic import ui

if __name__ == "__main__":
    print("启动 UI 测试...")
    print("功能:")
    print("  1. 左侧边栏顶部有绿色的 DEBUG 按钮")
    print("  2. 点击 DEBUG 按钮，侧边栏会向右展开")
    print("  3. 展开区域顶部有 'Debug 目录' 输入框")
    print("  4. 支持拖放文件夹到输入框")
    print("  5. 再次点击 DEBUG 按钮，侧边栏会收起")
    print()
    ui.main()
