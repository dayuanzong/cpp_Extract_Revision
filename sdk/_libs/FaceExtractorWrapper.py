import ctypes
import os
from pathlib import Path
import numpy as np

# Structure Definitions must match C++
class FaceInfo(ctypes.Structure):
    _fields_ = [
        ("jpg_data", ctypes.POINTER(ctypes.c_ubyte)),
        ("jpg_size", ctypes.c_int),
        ("landmarks", ctypes.c_float * 136),
        ("aligned_landmarks", ctypes.c_float * 136), # New field
        ("embedding_dim", ctypes.c_int),
        ("embedding", ctypes.c_float * 512),
        ("target_index", ctypes.c_int),
        ("target_sim", ctypes.c_float),
        ("is_target", ctypes.c_bool),
        ("source_rect", ctypes.c_float * 4),
        ("detect_score", ctypes.c_float),
        ("blur_variance", ctypes.c_float),
        ("pose_tag", ctypes.c_char * 32),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("blur_class", ctypes.c_int),
        ("mouth_value", ctypes.c_float),
        ("mouth_open", ctypes.c_bool),
        ("valid", ctypes.c_bool)
    ]

class FaceExtractorWrapper:
    def __init__(self, model_dir, device_id=0, s3fd_model_path=None, fan_model_path=None, rec_model_path=None):
        root_dir = Path(__file__).parent.parent.parent
        
        # Unified DLL path: always use bin directory
        self.lib_path = root_dir / "bin" / "FaceExtractorDLL.dll"

        if not self.lib_path.exists():
            raise FileNotFoundError(f"DLL not found at {self.lib_path}")

        try:
            # Add bin directory to DLL search path
            bin_dir = self.lib_path.parent
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            
            self.lib = ctypes.cdll.LoadLibrary(str(self.lib_path))
        except Exception as e:
            print(f"Failed to load DLL: {e}")
            raise RuntimeError(f"Could not load DLL from {self.lib_path}: {e}")
        
        # InitPipeline
        self.lib.InitPipeline.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
        self.lib.InitPipeline.restype = ctypes.c_int
        if hasattr(self.lib, "InitPipelineEx"):
            self.lib.InitPipelineEx.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
            self.lib.InitPipelineEx.restype = ctypes.c_int

        # ReleasePipeline
        self.lib.ReleasePipeline.argtypes = []
        self.lib.ReleasePipeline.restype = None

        # ProcessImage
        self.lib.ProcessImage.argtypes = [
            ctypes.c_wchar_p, 
            ctypes.POINTER(ctypes.POINTER(FaceInfo)), 
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int # face_type
        ]
        self.lib.ProcessImage.restype = ctypes.c_int

        if hasattr(self.lib, "ProcessImageMat"):
            self.lib.ProcessImageMat.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(FaceInfo)),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int
            ]
            self.lib.ProcessImageMat.restype = ctypes.c_int

        # FreeFaceResults
        self.lib.FreeFaceResults.argtypes = [
            ctypes.POINTER(FaceInfo),
            ctypes.c_int
        ]
        self.lib.FreeFaceResults.restype = None

        if hasattr(self.lib, "FreeImageBuffer"):
            self.lib.FreeImageBuffer.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
            self.lib.FreeImageBuffer.restype = None

        if hasattr(self.lib, "SetFilterParams"):
            self.lib.SetFilterParams.argtypes = [
                ctypes.c_int, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_float
            ]
            self.lib.SetFilterParams.restype = ctypes.c_int
        
        if hasattr(self.lib, "SetAlignSize"):
            self.lib.SetAlignSize.argtypes = [ctypes.c_int]
            self.lib.SetAlignSize.restype = ctypes.c_int
        
        if hasattr(self.lib, "SetMaxFaces"):
            self.lib.SetMaxFaces.argtypes = [ctypes.c_int]
            self.lib.SetMaxFaces.restype = ctypes.c_int
        
        if hasattr(self.lib, "SetJpegQuality"):
            self.lib.SetJpegQuality.argtypes = [ctypes.c_int]
            self.lib.SetJpegQuality.restype = ctypes.c_int

        if hasattr(self.lib, "GetEmbeddingDim"):
            self.lib.GetEmbeddingDim.argtypes = [ctypes.POINTER(ctypes.c_int)]
            self.lib.GetEmbeddingDim.restype = ctypes.c_int

        if hasattr(self.lib, "ExtractEmbedding"):
            self.lib.ExtractEmbedding.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int
            ]
            self.lib.ExtractEmbedding.restype = ctypes.c_int

        if hasattr(self.lib, "SetReferenceEmbeddings"):
            self.lib.SetReferenceEmbeddings.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float
            ]
            self.lib.SetReferenceEmbeddings.restype = ctypes.c_int
        if hasattr(self.lib, "ClearReferenceEmbeddings"):
            self.lib.ClearReferenceEmbeddings.argtypes = []
            self.lib.ClearReferenceEmbeddings.restype = ctypes.c_int

        if hasattr(self.lib, "InsertApp15Jpeg"):
            self.lib.InsertApp15Jpeg.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)
            ]
            self.lib.InsertApp15Jpeg.restype = ctypes.c_int

        if hasattr(self.lib, "ExtractApp15Jpeg"):
            self.lib.ExtractApp15Jpeg.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)
            ]
            self.lib.ExtractApp15Jpeg.restype = ctypes.c_int

        if hasattr(self.lib, "EmbeddingBestMatch"):
            self.lib.EmbeddingBestMatch.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
            ]
            self.lib.EmbeddingBestMatch.restype = ctypes.c_int

        if hasattr(self.lib, "GetVideoInfo"):
            self.lib.GetVideoInfo.argtypes = [
                ctypes.c_wchar_p,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]
            self.lib.GetVideoInfo.restype = ctypes.c_int

        if hasattr(self.lib, "ReadVideoFrame"):
            self.lib.ReadVideoFrame.argtypes = [
                ctypes.c_wchar_p, ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]
            self.lib.ReadVideoFrame.restype = ctypes.c_int

        model_dir_str = str(model_dir)
        use_custom = any([s3fd_model_path, fan_model_path, rec_model_path])
        if use_custom and hasattr(self.lib, "InitPipelineEx"):
            s3fd_val = str(s3fd_model_path) if s3fd_model_path else None
            fan_val = str(fan_model_path) if fan_model_path else None
            rec_val = str(rec_model_path) if rec_model_path else None
            ret = self.lib.InitPipelineEx(model_dir_str, device_id, s3fd_val, fan_val, rec_val)
        else:
            ret = self.lib.InitPipeline(model_dir_str, device_id)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize pipeline. Error code: {ret}")

    def __del__(self):
        if hasattr(self, 'lib'):
            self.lib.ReleasePipeline()

    def _parse_faces(self, out_faces, out_count):
        results = []
        try:
            count = out_count.value
            for i in range(count):
                face = out_faces[i]
                if not face.valid:
                    continue
                jpg_bytes = ctypes.string_at(face.jpg_data, face.jpg_size)
                landmarks_flat = list(face.landmarks)
                landmarks = [[landmarks_flat[2*j], landmarks_flat[2*j+1]] for j in range(68)]
                aligned_landmarks_flat = list(face.aligned_landmarks)
                aligned_landmarks = [[aligned_landmarks_flat[2*j], aligned_landmarks_flat[2*j+1]] for j in range(68)]
                emb_dim = int(face.embedding_dim)
                emb = None
                if emb_dim > 0:
                    emb = np.array(list(face.embedding)[:emb_dim], dtype=np.float32)
                target_index = int(face.target_index)
                target_sim = float(face.target_sim)
                is_target = bool(face.is_target)
                rect = list(face.source_rect)
                results.append({
                    "jpg_data": jpg_bytes,
                    "landmarks": landmarks,
                    "aligned_landmarks": aligned_landmarks,
                    "embedding": emb,
                    "target_index": target_index,
                    "target_sim": target_sim,
                    "is_target": is_target,
                    "rect": rect,
                    "score": float(face.detect_score),
                    "blur": face.blur_variance,
                    "blur_class": int(face.blur_class),
                    "pose": face.pose_tag.decode('utf-8', errors='ignore'),
                    "pitch": float(face.pitch),
                    "yaw": float(face.yaw),
                    "roll": float(face.roll),
                    "mouth_value": float(face.mouth_value),
                    "mouth_open": bool(face.mouth_open)
                })
        finally:
            if out_faces:
                self.lib.FreeFaceResults(out_faces, out_count)
        return results

    def process_image(self, img_path, face_type=2):
        """
        Process an image file and return extracted faces.
        Args:
            img_path: Path to image file (str or Path)
            face_type: Face type (int). Default 2 (FULL).
        Returns:
            list of dicts containing 'jpg_data', 'landmarks', 'rect', 'blur', 'pose'
        """
        out_faces = ctypes.POINTER(FaceInfo)()
        out_count = ctypes.c_int(0)
        
        ret = self.lib.ProcessImage(str(img_path), ctypes.byref(out_faces), ctypes.byref(out_count), int(face_type))
        
        if ret != 0:
            # print(f"ProcessImage failed for {img_path}. Code: {ret}")
            return []
            
        return self._parse_faces(out_faces, out_count)

    def process_frame(self, frame, face_type=2):
        if not hasattr(self.lib, "ProcessImageMat"):
            raise RuntimeError("ProcessImageMat 不可用，请重新编译 DLL")
        if frame is None:
            return []
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return []
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        h, w, _ = frame.shape
        step = int(frame.strides[0])
        out_faces = ctypes.POINTER(FaceInfo)()
        out_count = ctypes.c_int(0)
        ret = self.lib.ProcessImageMat(frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), w, h, step, ctypes.byref(out_faces), ctypes.byref(out_count), int(face_type))
        if ret != 0:
            return []
        return self._parse_faces(out_faces, out_count)

    def set_filter_params(self, enable_blur=1, blur_low=10.0, blur_high=20.0,
                          enable_pose=1, pitch_threshold=15.0, yaw_threshold=15.0,
                          enable_mouth=1, mouth_threshold=15.0):
        if not hasattr(self.lib, "SetFilterParams"):
            return 0
        return self.lib.SetFilterParams(int(enable_blur), float(blur_low), float(blur_high),
                                        int(enable_pose), float(pitch_threshold), float(yaw_threshold),
                                        int(enable_mouth), float(mouth_threshold))

    def set_align_size(self, size=256):
        if not hasattr(self.lib, "SetAlignSize"):
            return 0
        return self.lib.SetAlignSize(int(size))
    
    def set_max_faces(self, max_faces=0):
        if not hasattr(self.lib, "SetMaxFaces"):
            return 0
        return self.lib.SetMaxFaces(int(max_faces))

    def set_jpeg_quality(self, quality=90):
        if not hasattr(self.lib, "SetJpegQuality"):
            return 0
        return self.lib.SetJpegQuality(int(quality))

    def get_embedding_dim(self):
        if not hasattr(self.lib, "GetEmbeddingDim"):
            return 0
        dim = ctypes.c_int(0)
        ret = self.lib.GetEmbeddingDim(ctypes.byref(dim))
        if ret != 0:
            return 0
        return int(dim.value)

    def extract_embedding(self, img_bgr, landmarks):
        if not hasattr(self.lib, "ExtractEmbedding"):
            return None
        if img_bgr is None or landmarks is None:
            return None
        img = np.asarray(img_bgr, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            return None
        lmk = np.asarray(landmarks, dtype=np.float32).reshape(-1)
        if lmk.size < 136:
            return None
        emb_dim = self.get_embedding_dim()
        if emb_dim <= 0:
            return None
        out = np.zeros((emb_dim,), dtype=np.float32)
        ret = self.lib.ExtractEmbedding(
            img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            int(img.shape[1]), int(img.shape[0]), int(img.strides[0]),
            lmk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(lmk.size),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(emb_dim)
        )
        if ret != 0:
            return None
        return out

    def insert_app15_jpeg(self, jpg_bytes, app15_bytes):
        if not hasattr(self.lib, "InsertApp15Jpeg"):
            return None
        if jpg_bytes is None or app15_bytes is None:
            return None
        jpg_buf = ctypes.create_string_buffer(jpg_bytes)
        app_buf = ctypes.create_string_buffer(app15_bytes)
        out_data = ctypes.POINTER(ctypes.c_ubyte)()
        out_size = ctypes.c_int(0)
        ret = self.lib.InsertApp15Jpeg(
            ctypes.cast(jpg_buf, ctypes.POINTER(ctypes.c_ubyte)), int(len(jpg_bytes)),
            ctypes.cast(app_buf, ctypes.POINTER(ctypes.c_ubyte)), int(len(app15_bytes)),
            ctypes.byref(out_data), ctypes.byref(out_size)
        )
        if ret != 0 or not out_data or out_size.value <= 0:
            return None
        try:
            return ctypes.string_at(out_data, out_size.value)
        finally:
            if hasattr(self.lib, "FreeImageBuffer"):
                self.lib.FreeImageBuffer(out_data)

    def embedding_best_match(self, emb, refs):
        if not hasattr(self.lib, "EmbeddingBestMatch"):
            return -1, -1.0
        emb = np.asarray(emb, dtype=np.float32)
        refs = np.asarray(refs, dtype=np.float32)
        if emb.ndim != 1 or refs.ndim != 2:
            return -1, -1.0
        emb_dim = emb.shape[0]
        ref_count = refs.shape[0]
        best_idx = ctypes.c_int(-1)
        best_sim = ctypes.c_float(-1.0)
        ret = self.lib.EmbeddingBestMatch(emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), emb_dim,
                                          refs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ref_count,
                                          ctypes.byref(best_idx), ctypes.byref(best_sim))
        if ret != 0:
            return -1, -1.0
        return int(best_idx.value), float(best_sim.value)

    def set_reference_embeddings(self, refs, sim_threshold=0.4):
        if not hasattr(self.lib, "SetReferenceEmbeddings"):
            return False
        refs = np.asarray(refs, dtype=np.float32)
        if refs.ndim == 1:
            refs = np.expand_dims(refs, axis=0)
        if refs.ndim != 2:
            return False
        ref_count, ref_dim = refs.shape
        ret = self.lib.SetReferenceEmbeddings(refs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(ref_count), int(ref_dim), float(sim_threshold))
        return ret == 0

    def clear_reference_embeddings(self):
        if not hasattr(self.lib, "ClearReferenceEmbeddings"):
            return False
        ret = self.lib.ClearReferenceEmbeddings()
        return ret == 0

    def get_video_info(self, video_path):
        if not hasattr(self.lib, "GetVideoInfo"):
            return None
        frame_count = ctypes.c_int(0)
        fps = ctypes.c_double(0.0)
        width = ctypes.c_int(0)
        height = ctypes.c_int(0)
        ret = self.lib.GetVideoInfo(str(video_path), ctypes.byref(frame_count), ctypes.byref(fps), ctypes.byref(width), ctypes.byref(height))
        if ret != 0:
            return None
        return {
            "frame_count": int(frame_count.value),
            "fps": float(fps.value),
            "width": int(width.value),
            "height": int(height.value)
        }

    def read_video_frame(self, video_path, frame_index):
        if not hasattr(self.lib, "ReadVideoFrame"):
            return None
        out_data = ctypes.POINTER(ctypes.c_ubyte)()
        out_width = ctypes.c_int(0)
        out_height = ctypes.c_int(0)
        out_channels = ctypes.c_int(0)
        out_step = ctypes.c_int(0)
        ret = self.lib.ReadVideoFrame(str(video_path), int(frame_index), ctypes.byref(out_data),
                                      ctypes.byref(out_width), ctypes.byref(out_height),
                                      ctypes.byref(out_channels), ctypes.byref(out_step))
        if ret != 0:
            return None
        size = int(out_step.value) * int(out_height.value)
        buf = ctypes.string_at(out_data, size)
        if hasattr(self.lib, "FreeImageBuffer"):
            self.lib.FreeImageBuffer(out_data)
        arr = np.frombuffer(buf, dtype=np.uint8)
        frame = arr.reshape((int(out_height.value), int(out_width.value), int(out_channels.value)))
        return frame
