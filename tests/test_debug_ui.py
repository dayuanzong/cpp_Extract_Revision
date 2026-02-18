"""Debug UI 测试"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "sdk"))

from core_logic.ui import ExtractUI

def main():
    print("=" * 70)
    print("Debug UI 测试")
    print("=" * 70)
    print()
    print("测试内容:")
    print("  1. 缩略图布局：图片列 + 文件名列")
    print("  2. 窗口自动调整（仅在预览区域不足时）")
    print("  3. 调整后图片正确显示")
    print()
    print("测试步骤:")
    print("  1. 点击 Debug 按钮")
    print("  2. 输入测试目录")
    print("  3. 点击缩略图查看预览")
    print("  4. 观察窗口调整和图片显示")
    print()
    print("=" * 70)
    
    app = ExtractUI()
    app.mainloop()

if __name__ == "__main__":
    main()
