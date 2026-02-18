"""
测试 Debug UI - 图片预览和 Landmark 绘制
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "sdk"))

import tkinter as tk
from core_logic.ui import ExtractUI

def test_debug_preview():
    """测试 Debug UI 的图片预览和 Landmark 绘制功能"""
    print("=" * 70)
    print("Debug UI 测试 - 图片预览和 Landmark 绘制")
    print("=" * 70)
    print()
    print("新功能:")
    print("  • 缩略图以列的形式显示（单列）")
    print("  • 点击缩略图在右侧显示大图")
    print("  • 顶部显示图片名称")
    print("  • 底部3个按钮切换显示模式:")
    print("    - Points: 显示关键点和索引")
    print("    - Lines: 显示连线")
    print("    - Mesh: 显示网格")
    print("  • 默认显示 Points 模式")
    print()
    print("测试步骤:")
    print("  1. 点击左侧边栏的 Debug 按钮")
    print("  2. 输入包含 DFL 人脸图片的目录")
    print("  3. 缩略图以单列形式显示在左侧")
    print("  4. 点击任意缩略图")
    print("  5. 右侧显示大图和 landmarks（Points 模式）")
    print("  6. 点击 Lines 或 Mesh 按钮切换显示模式")
    print()
    print("注意: 需要图片包含 DFL 数据才能显示 landmarks")
    print()
    print("按 Ctrl+C 或关闭窗口退出测试")
    print("=" * 70)
    print()
    
    app = ExtractUI()
    app.title("Debug UI 测试 - 图片预览和 Landmark 绘制")
    app.geometry("1200x800")
    
    # 在控制台输出提示
    app.log("=" * 60, "info")
    app.log("Debug UI 测试 - 图片预览和 Landmark 绘制", "info")
    app.log("=" * 60, "info")
    app.log("", "info")
    app.log("新特性:", "info")
    app.log("  • 单列缩略图显示", "info")
    app.log("  • 点击预览大图", "info")
    app.log("  • 3种 Landmark 显示模式", "info")
    app.log("", "info")
    app.log("提示: 输入包含 DFL 人脸图片的目录", "info")
    app.log("例如: 已提取的人脸图片目录", "info")
    
    app.mainloop()

if __name__ == "__main__":
    test_debug_preview()
