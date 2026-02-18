import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def main():
    script_dir = Path(__file__).parent
    deps_dir = script_dir / "deps"
    deps_dir.mkdir(exist_ok=True)

    # 1. ONNX Runtime
    # GPU Version 1.18.1
    onnx_version = "1.18.1"
    onnx_url = f"https://github.com/microsoft/onnxruntime/releases/download/v{onnx_version}/onnxruntime-win-x64-gpu-{onnx_version}.zip"
    onnx_zip = deps_dir / f"onnxruntime-win-x64-gpu-{onnx_version}.zip"
    onnx_extract_dir = deps_dir / "onnxruntime"

    if not onnx_extract_dir.exists():
        if not onnx_zip.exists():
            if not download_file(onnx_url, onnx_zip):
                print("Failed to download ONNX Runtime.")
                return

        if extract_zip(onnx_zip, deps_dir):
            # Rename folder to simplified name
            extracted_folder = deps_dir / f"onnxruntime-win-x64-gpu-{onnx_version}"
            if extracted_folder.exists():
                extracted_folder.rename(onnx_extract_dir)
            else:
                print(f"Warning: Expected extracted folder {extracted_folder} not found.")

    # 2. OpenCV
    # Try to find a ZIP release from thommyho/Cpp-OpenCV-Windows-PreBuilts
    # Since we can't easily parse GitHub releases page without HTML parsing, 
    # we'll try a few known versions.
    # v4.8.0, v4.5.5
    
    opencv_extract_dir = deps_dir / "opencv"
    if not opencv_extract_dir.exists():
        opencv_versions = ["4.8.0", "4.5.5"]
        success = False
        
        for version in opencv_versions:
            # Construct URL (guessing format based on typical releases)
            # Actually, looking at the search result, thommyho uses releases.
            # But the artifact name might vary.
            # Let's try to use the official SourceForge link if possible, but that's an EXE (7z).
            # If we can't unzip 7z, we are stuck with exe.
            # But wait, we can try to install 'py7zr' via pip.
            
            try:
                import py7zr
                print("py7zr is available.")
                use_7z = True
            except ImportError:
                print("py7zr not found. Attempting to install...")
                import subprocess
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "py7zr"])
                    import py7zr
                    use_7z = True
                except Exception as e:
                    print(f"Failed to install py7zr: {e}")
                    use_7z = False

            if use_7z:
                # Download official OpenCV exe (7z)
                opencv_url = f"https://github.com/opencv/opencv/releases/download/{version}/opencv-{version}-vc14_vc15.exe"
                opencv_exe = deps_dir / f"opencv-{version}.exe"
                
                if not opencv_exe.exists():
                    if not download_file(opencv_url, opencv_exe):
                        continue
                
                print(f"Extracting {opencv_exe} using py7zr...")
                try:
                    with py7zr.SevenZipFile(opencv_exe, mode='r') as z:
                        z.extractall(path=deps_dir)
                    # It usually extracts to 'opencv' folder directly
                    if (deps_dir / "opencv").exists():
                        success = True
                        break
                except Exception as e:
                    print(f"Error extracting with py7zr: {e}")
            else:
                print("Skipping OpenCV download as py7zr is not available and we need it for the official .exe (7z) archive.")
                # Could try to find a ZIP from third party, but URLs are less predictable.
                break
        
        if not success:
            print("Failed to set up OpenCV.")
            print("Please manually download OpenCV 4.x (vc14_vc15) and extract it to cpp_Extract_Revision/deps/opencv")

if __name__ == "__main__":
    main()
