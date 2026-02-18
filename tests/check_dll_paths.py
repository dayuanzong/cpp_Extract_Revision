"""
检查DLL路径配置是否正确
"""
import sys
from pathlib import Path

def check_dll_paths():
    """检查所有DLL路径配置"""
    
    root_dir = Path(__file__).parent.parent
    print("=" * 60)
    print("DLL路径配置检查")
    print("=" * 60)
    
    # 检查bin目录
    bin_dir = root_dir / "bin"
    print(f"\n1. 检查bin目录: {bin_dir}")
    
    if not bin_dir.exists():
        print("   ❌ bin目录不存在！")
        print("   请先运行 core/build_cpp.bat 构建项目")
        return False
    else:
        print("   ✓ bin目录存在")
    
    # 检查必需的DLL文件
    required_dlls = [
        "FaceExtractorDLL.dll",
        "opencv_world455.dll",
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll"
    ]
    
    print("\n2. 检查必需的DLL文件:")
    all_exist = True
    for dll_name in required_dlls:
        dll_path = bin_dir / dll_name
        if dll_path.exists():
            size_mb = dll_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {dll_name} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {dll_name} 不存在")
            all_exist = False
    
    # 检查可选的DLL
    optional_dlls = [
        "FaceExtractor.exe",
        "onnxruntime_providers_tensorrt.dll"
    ]
    
    print("\n3. 检查可选文件:")
    for dll_name in optional_dlls:
        dll_path = bin_dir / dll_name
        if dll_path.exists():
            size_mb = dll_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {dll_name} ({size_mb:.2f} MB)")
        else:
            print(f"   - {dll_name} 不存在（可选）")
    
    # 检查Python代码中的路径配置
    print("\n4. 检查Python代码配置:")
    
    # 检查FaceExtractorWrapper.py
    wrapper_file = root_dir / "sdk" / "_libs" / "FaceExtractorWrapper.py"
    if wrapper_file.exists():
        content = wrapper_file.read_text(encoding='utf-8')
        if 'bin" / "FaceExtractorDLL.dll' in content or 'bin/FaceExtractorDLL.dll' in content:
            print("   ✓ FaceExtractorWrapper.py 使用bin目录")
        else:
            print("   ❌ FaceExtractorWrapper.py 未使用bin目录")
            all_exist = False
    else:
        print("   ❌ FaceExtractorWrapper.py 不存在")
        all_exist = False
    
    # 检查main.py
    main_file = root_dir / "sdk" / "main.py"
    if main_file.exists():
        content = main_file.read_text(encoding='utf-8')
        if 'bin_dir' in content:
            print("   ✓ main.py 配置了bin_dir")
        else:
            print("   ⚠ main.py 可能未正确配置bin_dir")
    else:
        print("   ❌ main.py 不存在")
        all_exist = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_exist:
        print("✓ 所有检查通过！DLL路径配置正确")
        print("\n可以运行以下命令测试:")
        print("  cd tests")
        print("  python verify_fix.py")
        print("  python test_face_proportion.py")
    else:
        print("❌ 存在配置问题，请检查上述错误")
        print("\n建议操作:")
        print("  1. 运行 core\\build_cpp.bat 重新构建")
        print("  2. 确保所有Python文件已更新")
    print("=" * 60)
    
    return all_exist

if __name__ == "__main__":
    success = check_dll_paths()
    sys.exit(0 if success else 1)
