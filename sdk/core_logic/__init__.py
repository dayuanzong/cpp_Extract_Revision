
import logging
import os
import sys
from pathlib import Path

# Ensure _libs is in path for all modules in core_logic
def setup_libs_path():
    # Logic to find _libs relative to this file or the executable
    # If running from source: ../../_libs
    # If frozen: sys._MEIPASS/_libs (handled by PyInstaller usually, but we might need explicit add)
    
    current_file = Path(__file__).resolve()
    
    # Assuming standard structure:
    # root/
    #   core_logic/
    #   _libs/
    
    root_dir = current_file.parent.parent
    libs_dir = root_dir / "_libs"
    ort_dir = root_dir / "onnxruntime" / "capi"
    
    if libs_dir.exists():
        if str(libs_dir) not in sys.path:
            sys.path.insert(0, str(libs_dir))

    if ort_dir.exists():
        os.environ["PATH"] = str(ort_dir) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(ort_dir))
        except Exception:
            pass
    
    # Also handle frozen state where _libs might be in a temp dir
    if getattr(sys, 'frozen', False):
        # PyInstaller
        if hasattr(sys, '_MEIPASS'):
            libs_frozen = Path(sys._MEIPASS) / "_libs"
            if libs_frozen.exists() and str(libs_frozen) not in sys.path:
                sys.path.insert(0, str(libs_frozen))
            ort_frozen = Path(sys._MEIPASS) / "onnxruntime" / "capi"
            if ort_frozen.exists():
                os.environ["PATH"] = str(ort_frozen) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(ort_frozen))
                except Exception:
                    pass
        else:
            # Nuitka standalone
            base_dir = Path(sys.executable).parent
            possible_dirs = [
                base_dir / "_libs",
                base_dir / "TF_Extract.dist" / "_libs",
                base_dir / "main.dist" / "_libs"
            ]
            for p in possible_dirs:
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                    break

setup_libs_path()
