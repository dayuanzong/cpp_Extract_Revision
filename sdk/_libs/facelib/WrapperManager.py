from pathlib import Path
import sys

# Try to import FaceExtractorWrapper
# Since _libs is in sys.path, we can import it directly
try:
    from FaceExtractorWrapper import FaceExtractorWrapper
except ImportError:
    # If run from within package structure without sys.path setup
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        from FaceExtractorWrapper import FaceExtractorWrapper
    except ImportError:
        print("Warning: FaceExtractorWrapper not found.")
        FaceExtractorWrapper = None

_wrapper_instance = None
_wrapper_device_id = None

def get_wrapper(device_id=0):
    global _wrapper_instance
    global _wrapper_device_id
    
    if _wrapper_instance is None:
        if FaceExtractorWrapper is None:
            raise ImportError("FaceExtractorWrapper module is missing")

        # Determine model_dir
        # Structure:
        # root/
        #   assets/
        #     models/ (was models_exported)
        #   sdk/
        #     _libs/
        #       facelib/
        
        # sdk/_libs/facelib -> sdk/_libs -> sdk -> root
        root_dir = Path(__file__).parent.parent.parent.parent
        model_dir = root_dir / "assets" / "models"
        
        if not model_dir.exists():
            # Try old name if new one doesn't exist
            model_dir = root_dir / "models_exported"
            
        if not model_dir.exists():
            # Try alternate location if running from dist
            model_dir = root_dir / "TF_Extract.dist" / "models_exported"
            
        if not model_dir.exists():
             # Fallback to local models dir if present
             model_dir = Path("models_exported").absolute()

        if not model_dir.exists():
            print(f"Warning: Model directory not found at {model_dir}. Wrapper might fail if models are needed.")
            
        _wrapper_instance = FaceExtractorWrapper(model_dir, device_id)
        _wrapper_device_id = device_id
        
    else:
        # Check if requested device matches initialized device
        # Note: FaceExtractorWrapper is a singleton-like resource holder for the DLL.
        # If we need to support multiple devices, we might need multiple instances or a way to switch context.
        # For now, we assume one device per process.
        if _wrapper_device_id != device_id:
             print(f"Warning: Wrapper already initialized with device {_wrapper_device_id}, but requested {device_id}. Ignoring request and using existing instance.")
             
    return _wrapper_instance
