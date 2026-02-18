"""
测试 Debug UI - TurboJPEG 虚拟滚动版本
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "sdk"))

import tkinter as tk
from core_logic.ui import ExtractUI

def test_debug_turbojpeg():
    """测试 Debug UI 的 TurboJPEG 虚拟滚动功能"""
    print("=" * 70)
    print("Debug UI 测试 - TurboJPEG 虚拟滚动版本")
    print("=" * 70)
    print()
    print("新功能:")
    print("  • 使用 TurboJPEG 加载图片（移除 Pillow 依赖）")
    print("  • 48x48 像素缩略图")
    print("  • 虚拟滚动（只渲染可见区域）")
    print("  • 支持大量图片（不限制数量）")
    print("  • 滚动条支持上下滑动")
    print("  • 鼠标滚轮支持")
    print()
    print("测试步骤:")
    print("  1. 点击左侧边栏的 Debug 按钮")
    print("  2. 输入或拖放包含大量图片的文件夹")
    print("  3. 观察缩略图自动加载（48x48）")
    print("  4. 使用滚动条或鼠标滚轮查看更多图片")
    print("  5. 测试两个目录的对比功能")
    print()
    print("按 Ctrl+C 或关闭窗口退出测试")
    print("=" * 70)
    print()
    
    app = ExtractUI()
    app.title("Debug UI 测试 - TurboJPEG 虚拟滚动")
    app.geometry("1200x800")
    
    # 在控制台输出提示
    app.log("=" * 60, "info")
    app.log("Debug UI 测试 - TurboJPEG 虚拟滚动", "info")
    app.log("=" * 60, "info")
    app.log("", "info")
    app.log("新特性:", "info")
    app.log("  • TurboJPEG 快速解码", "info")
    app.log("  • 48x48 缩略图", "info")
    app.log("  • 虚拟滚动（只渲染可见部分）", "info")
    app.log("  • 支持大量图片", "info")
    app.log("", "info")
    app.log("提示: 输入包含大量图片的目录测试性能", "info")
    
    app.mainloop()

if __name__ == "__main__":
    test_debug_turbojpeg()
