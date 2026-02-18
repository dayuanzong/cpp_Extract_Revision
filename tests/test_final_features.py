"""测试最终功能"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "sdk"))

from core_logic.ui import ExtractUI

def main():
    print("=" * 70)
    print("Debug UI 最终功能测试")
    print("=" * 70)
    print()
    print("测试功能:")
    print("  1. 缩略图选中状态（红框）")
    print("  2. ALT+滚轮缩放预览图")
    print("  3. 鼠标拖动预览图")
    print("  4. 滚动条拖动到底部显示最后一张")
    print()
    print("测试步骤:")
    print("  1. 点击 Debug 按钮")
    print("  2. 输入测试目录")
    print("  3. 点击缩略图，观察红框")
    print("  4. 拖动滚动条到底部，确认显示最后一张")
    print("  5. ALT+滚轮放大预览图")
    print("  6. 鼠标拖动预览图查看不同区域")
    print()
    print("=" * 70)
    
    app = ExtractUI()
    app.mainloop()

if __name__ == "__main__":
    main()
