
import argparse
import multiprocessing
import os
from pathlib import Path
import sys

def setup_env():
    # If frozen, sys.executable is the app.
    # If script, __file__ is the script.
    
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        base_dir = Path(sys.executable).parent
        # If PyInstaller one-file, sys._MEIPASS is the temp dir
        if hasattr(sys, '_MEIPASS'):
             libs_dir = Path(sys._MEIPASS) / "_libs"
             bin_dir = Path(sys._MEIPASS) / "bin"
        else:
             # Nuitka standalone: check potential locations
             # 1. Next to EXE
             # 2. Inside *.dist folder (e.g. TF_Extract.dist)
             possible_dirs = [
                 base_dir / "_libs",
                 base_dir / "TF_Extract.dist" / "_libs",
                 base_dir / "main.dist" / "_libs"
             ]
             libs_dir = base_dir / "_libs" # default
             for p in possible_dirs:
                 if p.exists():
                     libs_dir = p
                     break
                     
             bin_dir = base_dir / "bin"
    else:
        # Running as script
        base_dir = Path(__file__).parent.resolve()
        libs_dir = base_dir / "_libs"
        root_dir = base_dir.parent
        bin_dir = root_dir / "bin"

    if libs_dir.exists():
        if str(libs_dir) not in sys.path:
            sys.path.insert(0, str(libs_dir))

    # Unified DLL path: all DLLs are in bin directory
    if bin_dir.exists():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(bin_dir))
        except Exception:
            pass
            
    # Also ensure models dir is accessible if needed, though we use absolute paths or relative to libs usually.

# Run setup_env unconditionally to ensure child processes (multiprocessing spawn) get the path.
setup_env()

if __name__ == '__main__':
    # On Windows, calling freeze_support() is necessary for multiprocessing to work
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--selftest-onnxruntime", action="store_true")
    args, _ = parser.parse_known_args()
    if args.selftest_onnxruntime:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print("ONNXRuntime 自检成功")
            print(f"Providers: {providers}")
            sys.exit(0)
        except Exception as e:
            print(f"ONNXRuntime 自检失败: {e}")
            sys.exit(1)
    
    # Now we can import ui
    try:
        from core_logic import ui
        ui.main()
    except ImportError as e:
        print(f"Error importing UI: {e}")
        # If running from source, make sure parent dir is in path or we run as module
        # If we run `python main.py`, `core_logic` must be importable.
        # Since main.py is in root, and core_logic is a subdir with __init__.py, it should work.
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
