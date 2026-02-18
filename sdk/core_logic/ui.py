
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import threading
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

# Try to import TkinterDnD for drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND_AVAILABLE = True
    TkBase = TkinterDnD.Tk
except ImportError:
    TKDND_AVAILABLE = False
    TkBase = tk.Tk

from . import extractor
from . import filter
from . import utils
from . import debug_backend

DEFAULT_CONFIG = {
    "ex_input": "",
    "ex_output": "",
    "face_type": "whole_face",
    "img_size": 1024,
    "gpu_idx_str": "GPU 0",
    "loader_threads": 1,
    "saver_threads": 4,
    "jpeg_q": 100,
    "debug": False,
    "s3fd_model_path": "",
    "fan_model_path": "",
    "rec_model_path": "",
    "use_target": True,
    "ref_dir": "",
    "sim_thresh": 0.4,
    "multi_target": False,
    "keep_non_target": False,
    "reduce_non_target": False,
    "non_target_keep_interval": 5,
    "auto_augment": False,
    "skip_start": 0.0,
    "skip_end": 0.0,
    "step": 0,
    "filter_input": "",
    "filter_output": "",
    "filter_mode": "pose",
    "filter_ref_dir": "",
    "filter_sim_thresh": 0.4,
    "filter_multi_target": False,
    "pose_single": False,
    "pitch_thresh": 10.0,
    "yaw_thresh": 45.0,
    "mouth_thresh": 15.0,
    "blur_low": 10.0,
    "blur_high": 20.0,
    "rt_blur": False,
    "rt_blur_low": 10.0,
    "rt_blur_high": 20.0,
    "rt_mouth": False,
    "rt_mouth_thresh": 15.0,
    "rt_pose": False,
    "rt_pose_dir": "抬头",
    "rt_pitch": 10.0,
    "rt_yaw": 45.0,
    "save_no_face": True
}

class Tooltip:
    def __init__(self, widget, text, delay=700):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self.tipwindow = None
        self.widget.bind("<Enter>", self._schedule, add='+')
        self.widget.bind("<Leave>", self._hide, add='+')
        self.widget.bind("<ButtonPress>", self._hide, add='+')

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass
        label = tk.Label(tw, text=self.text, justify='left', background='#ffffe0', foreground='#333333', relief='solid', borderwidth=1, font=("Microsoft YaHei UI", 9))
        label.pack(ipadx=6, ipady=4)
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self, event=None):
        self._cancel()
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

class ExtractUI(TkBase):
    def __init__(self):
        super().__init__()
        
        self.title("TensorFlow Extract Tool (Optimized)")
        self.geometry("800x850")
        
        self.process = None
        # Use multiprocessing Queue for IPC
        self.log_queue = multiprocessing.Queue()
        self.last_msg_was_progress = False
        self.config = dict(DEFAULT_CONFIG)
        self.config_path = self.get_config_path()
        self.load_persisted_config()
        self._tooltips = []
        self.log_file = None
        self.log_file_path = None
        
        self.save_no_face_var = tk.BooleanVar(value=self.get_conf("save_no_face", False))
        
        self.create_widgets()
        self.init_log_file()
        self.default_config = dict(DEFAULT_CONFIG)
        self.check_queue()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.stop_process()
        self.save_persisted_config()
        self.close_log_file()
        self.destroy()

    def init_log_file(self):
        root_dir = Path(__file__).parent.parent.parent
        logs_dir = root_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            for p in logs_dir.glob("*.log"):
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        self.log_file_path = logs_dir / "run.log"
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        self.log(f"日志文件: {self.log_file_path}", "info")

    def close_log_file(self):
        if self.log_file:
            try:
                self.log_file.flush()
                self.log_file.close()
            finally:
                self.log_file = None

    def get_conf(self, key, default):
        return self.config.get(key, default)

    def get_config_path(self):
        base_dir = os.environ.get("APPDATA") or os.path.expanduser("~")
        return Path(base_dir) / "TensorFlow_Extract" / "ui_state.json"

    def load_persisted_config(self):
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.config.update(data)
        except Exception:
            pass

    def save_persisted_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = self.get_current_config()
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_tooltip(self, widget, text):
        self._tooltips.append(Tooltip(widget, text))

    def get_current_config(self):
        return {
            "ex_input": self.ex_input_var.get(),
            "ex_output": self.ex_output_var.get(),
            "face_type": self.face_type_var.get(),
            "img_size": self.img_size_var.get(),
            "gpu_idx_str": self.gpu_idx_var.get(),
            "loader_threads": self.loader_threads_var.get(),
            "saver_threads": self.saver_threads_var.get(),
            "jpeg_q": self.jpeg_q_var.get(),
            "debug": self.debug_var.get(),
            "s3fd_model_path": "" if self.s3fd_model_var.get() in ["", "默认"] else self.s3fd_model_var.get(),
            "fan_model_path": "" if self.fan_model_var.get() in ["", "默认"] else self.fan_model_var.get(),
            "rec_model_path": "" if self.rec_model_var.get() in ["", "默认"] else self.rec_model_var.get(),
            "use_target": self.use_target_var.get(),
            "ref_dir": self.ref_dir_var.get(),
            "sim_thresh": self.sim_thresh_var.get(),
            "multi_target": self.multi_target_var.get(),
            "keep_non_target": self.keep_non_target_var.get(),
            "reduce_non_target": self.reduce_non_target_var.get(),
            "non_target_keep_interval": self.non_target_keep_interval_var.get(),
            "auto_augment": self.auto_augment_var.get(),
            "skip_start": self.skip_start_var.get(),
            "skip_end": self.skip_end_var.get(),
            "step": self.step_var.get(),
            "filter_input": self.filter_input_var.get(),
            "filter_output": self.filter_output_var.get(),
            "filter_mode": self.filter_mode_var.get(),
            "filter_ref_dir": self.filter_ref_dir_var.get(),
            "filter_sim_thresh": self.filter_sim_thresh_var.get(),
            "filter_multi_target": self.filter_multi_target_var.get(),
            "filter_target_pick": self.filter_target_pick_var.get(),
            "pose_single": self.pose_var.get(),
            "pitch_thresh": self.pitch_var.get(),
            "yaw_thresh": self.yaw_var.get(),
            "mouth_thresh": self.mouth_thresh_var.get(),
            "blur_low": self.blur_low_var.get(),
            "blur_high": self.blur_high_var.get(),
            "rt_blur": self.rt_blur_var.get(),
            "rt_blur_low": self.rt_blur_low_var.get(),
            "rt_blur_high": self.rt_blur_high_var.get(),
            "rt_mouth": self.rt_mouth_var.get(),
            "rt_mouth_thresh": self.rt_mouth_thresh_var.get(),
            "rt_pose": self.rt_pose_var.get(),
            "rt_pose_dir": self.rt_pose_dir_var.get(),
            "rt_pitch": self.rt_pitch_var.get(),
            "rt_yaw": self.rt_yaw_var.get(),
            "save_no_face": self.save_no_face_var.get()
        }

    def apply_config(self, config):
        self.ex_input_var.set(config.get("ex_input", ""))
        self.ex_output_var.set(config.get("ex_output", ""))
        self.face_type_var.set(config.get("face_type", "whole_face"))
        self.img_size_var.set(config.get("img_size", 1024))
        self.gpu_idx_var.set(config.get("gpu_idx_str", "GPU 0"))
        self.loader_threads_var.set(config.get("loader_threads", min(4, multiprocessing.cpu_count())))
        self.saver_threads_var.set(config.get("saver_threads", max(1, multiprocessing.cpu_count() - 2)))
        self.jpeg_q_var.set(config.get("jpeg_q", 100))
        self.debug_var.set(config.get("debug", False))
        s3fd_val = config.get("s3fd_model_path", "")
        fan_val = config.get("fan_model_path", "")
        rec_val = config.get("rec_model_path", "")
        self.s3fd_model_var.set(s3fd_val if s3fd_val else "默认")
        self.fan_model_var.set(fan_val if fan_val else "默认")
        self.rec_model_var.set(rec_val if rec_val else "默认")
        self.use_target_var.set(config.get("use_target", False))
        self.ref_dir_var.set(config.get("ref_dir", ""))
        self.sim_thresh_var.set(config.get("sim_thresh", 0.4))
        self.multi_target_var.set(config.get("multi_target", False))
        self.keep_non_target_var.set(config.get("keep_non_target", False))
        self.reduce_non_target_var.set(config.get("reduce_non_target", False))
        self.non_target_keep_interval_var.set(config.get("non_target_keep_interval", 5))
        self.auto_augment_var.set(config.get("auto_augment", False))
        self.skip_start_var.set(config.get("skip_start", 0.0))
        self.skip_end_var.set(config.get("skip_end", 0.0))
        self.step_var.set(config.get("step", 1))
        self.filter_input_var.set(config.get("filter_input", ""))
        self.filter_output_var.set(config.get("filter_output", ""))
        self.filter_mode_var.set(config.get("filter_mode", "pose"))
        self.filter_ref_dir_var.set(config.get("filter_ref_dir", ""))
        self.filter_sim_thresh_var.set(config.get("filter_sim_thresh", 0.4))
        self.filter_multi_target_var.set(config.get("filter_multi_target", False))
        self.filter_target_pick_var.set(config.get("filter_target_pick", True))
        self.pose_var.set(config.get("pose_single", False))
        self.pitch_var.set(config.get("pitch_thresh", 15.0))
        self.yaw_var.set(config.get("yaw_thresh", 20.0))
        self.mouth_thresh_var.set(config.get("mouth_thresh", 20.0))
        self.blur_low_var.set(config.get("blur_low", 100.0))
        self.blur_high_var.set(config.get("blur_high", 500.0))
        self.rt_blur_var.set(config.get("rt_blur", False))
        self.rt_blur_low_var.set(config.get("rt_blur_low", 10.0))
        self.rt_blur_high_var.set(config.get("rt_blur_high", 20.0))
        self.rt_mouth_var.set(config.get("rt_mouth", False))
        self.rt_mouth_thresh_var.set(config.get("rt_mouth_thresh", 15.0))
        self.rt_pose_var.set(config.get("rt_pose", False))
        self.rt_pose_dir_var.set(config.get("rt_pose_dir", "抬头"))
        self.rt_pitch_var.set(config.get("rt_pitch", 10.0))
        self.rt_yaw_var.set(config.get("rt_yaw", 45.0))
        self.save_no_face_var.set(config.get("save_no_face", False))
        self.toggle_target_options()
        self.refresh_target_sidebar()
        self.on_rt_pose_toggle()
        self.update_filter_ui()

    def reset_to_defaults(self):
        if not hasattr(self, "default_config"):
            self.default_config = self.get_current_config()
        self.apply_config(self.default_config)

    def create_widgets(self):
        style = ttk.Style()
        try:
            style.configure("TPanedwindow", sashwidth=4, sashrelief='flat')
            style.configure('Vertical.TButton', font=('Arial', 9))
        except:
            pass
        
        self.main_h_container = tk.Frame(self)
        self.main_h_container.pack(fill='both', expand=True)

        # 抽屉式侧边栏容器
        self.drawer_container = tk.Frame(self.main_h_container, bg='#f0f0f0')
        self.drawer_container.place(x=0, y=0, relheight=1.0, width=32)
        
        # 侧边栏主框架（固定32px宽度）
        self.sidebar_frame = tk.Frame(self.drawer_container, width=32, bg='#f0f0f0', relief='sunken', bd=1)
        self.sidebar_frame.pack(side='left', fill='y')
        self.sidebar_frame.pack_propagate(False)
        
        # 抽屉内容框架（可展开部分）
        self.drawer_content = tk.Frame(self.drawer_container, bg='#ffffff', relief='raised', bd=2)
        self.drawer_content.pack(side='left', fill='both', expand=True)
        self.drawer_content.pack_propagate(False)
        
        # 抽屉状态
        self.drawer_expanded = False
        self.drawer_width = 0  # 当前抽屉宽度
        self.drawer_target_width = 0  # 目标宽度
        self.drawer_animation_id = None
        
        # Debug 区域（在抽屉内容中）
        self.debug_frame = tk.Frame(self.drawer_content, bg='#ffffff')
        self.debug_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 输入框区域（顶部）
        input_frame = tk.Frame(self.debug_frame, bg='#ffffff')
        input_frame.pack(fill='x', pady=(0, 10))
        
        # 第一个目录输入框
        self.debug_dir1_var = tk.StringVar()
        self.debug_dir1_var.trace('w', lambda *args: self.on_debug_dir_changed())
        dir1_frame = tk.Frame(input_frame, bg='#ffffff')
        dir1_frame.pack(fill='x', pady=(0, 5))
        
        dir1_label = tk.Label(dir1_frame, text="目录 1:", bg='#ffffff', font=("Microsoft YaHei UI", 9), width=6, anchor='w')
        dir1_label.pack(side='left')
        
        dir1_entry_frame = tk.Frame(dir1_frame, bg='#ffffff')
        dir1_entry_frame.pack(side='left', fill='x', expand=True)
        
        self.debug_dir1_entry = ttk.Entry(dir1_entry_frame, textvariable=self.debug_dir1_var)
        self.debug_dir1_entry.pack(side='left', fill='x', expand=True)
        
        dir1_browse_btn = ttk.Button(dir1_entry_frame, text="...", width=3,
                                     command=lambda: self.browse_path(self.debug_dir1_var, False))
        dir1_browse_btn.pack(side='left', padx=(2, 0))
        
        # 第二个目录输入框
        self.debug_dir2_var = tk.StringVar()
        self.debug_dir2_var.trace('w', lambda *args: self.on_debug_dir_changed())
        dir2_frame = tk.Frame(input_frame, bg='#ffffff')
        dir2_frame.pack(fill='x')
        
        dir2_label = tk.Label(dir2_frame, text="目录 2:", bg='#ffffff', font=("Microsoft YaHei UI", 9), width=6, anchor='w')
        dir2_label.pack(side='left')
        
        dir2_entry_frame = tk.Frame(dir2_frame, bg='#ffffff')
        dir2_entry_frame.pack(side='left', fill='x', expand=True)
        
        self.debug_dir2_entry = ttk.Entry(dir2_entry_frame, textvariable=self.debug_dir2_var)
        self.debug_dir2_entry.pack(side='left', fill='x', expand=True)
        
        dir2_browse_btn = ttk.Button(dir2_entry_frame, text="...", width=3,
                                     command=lambda: self.browse_path(self.debug_dir2_var, False))
        dir2_browse_btn.pack(side='left', padx=(2, 0))
        
        # 添加拖放支持
        if TKDND_AVAILABLE:
            for entry, var in [(self.debug_dir1_entry, self.debug_dir1_var), 
                              (self.debug_dir2_entry, self.debug_dir2_var)]:
                entry.drop_target_register(DND_FILES)
                def make_drop_handler(target_var):
                    def on_drop(event):
                        data = event.data
                        if data.startswith('{') and data.endswith('}'):
                            data = data[1:-1]
                        data = data.strip('"').strip("'")
                        path = Path(data)
                        if path.exists():
                            if path.is_dir():
                                target_var.set(str(path).replace('/', '\\'))
                            elif path.is_file():
                                target_var.set(str(path.parent).replace('/', '\\'))
                    return on_drop
                entry.dnd_bind('<<Drop>>', make_drop_handler(var))
        
        # 排序对比勾选框
        self.sort_compare_var = tk.BooleanVar(value=False)
        sort_check = ttk.Checkbutton(input_frame, text="名称排序对比", variable=self.sort_compare_var,
                                     command=self.on_debug_dir_changed)
        sort_check.pack(fill='x', pady=(5, 0))
        
        # 缩略图显示区域（两列显示）
        self.thumbnail_canvas = tk.Canvas(self.debug_frame, bg='#f5f5f5', highlightthickness=1, highlightbackground='#cccccc')
        self.thumbnail_canvas.pack(fill='both', expand=True)
        
        # 滚动条
        thumb_scrollbar = ttk.Scrollbar(self.thumbnail_canvas, orient='vertical', command=self._on_thumbnail_scroll)
        thumb_scrollbar.pack(side='right', fill='y')
        self.thumbnail_scrollbar = thumb_scrollbar
        
        # 缩略图容器框架（四列显示：图片1、文件名1、图片2、文件名2）
        self.thumbnail_frame = tk.Frame(self.thumbnail_canvas, bg='#f5f5f5')
        self.thumbnail_canvas_window = self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_frame, anchor='nw')
        
        # 预览相关变量
        self.current_preview_image = None  # 当前预览的图片路径
        self.current_preview_photo = None  # 当前预览的 PhotoImage
        self.landmark_mode = tk.StringVar(value='points')  # 默认 points 模式
        self.preview_zoom_scale = 1.0  # 预览图缩放比例
        self.selected_thumbnail_frame = None  # 当前选中的缩略图框架
        
        # 绑定事件
        self.thumbnail_canvas.bind('<Configure>', self._on_thumbnail_canvas_configure)
        self.thumbnail_canvas.bind('<MouseWheel>', self._on_thumbnail_mousewheel)
        
        # 初始化 TurboJPEG
        try:
            from turbojpeg import TurboJPEG
            dll_path = str(Path(__file__).parent.parent.parent / "bin" / "turbojpeg.dll")
            self.jpeg_decoder = TurboJPEG(dll_path)
            self.turbojpeg_available = True
        except Exception as e:
            self.jpeg_decoder = None
            self.turbojpeg_available = False
            print(f"TurboJPEG 初始化失败: {e}")
        
        # 虚拟滚动相关变量
        self.thumbnail_images = []  # 保持图片引用
        self.thumbnail_labels = []  # 标签引用
        self.all_image_files = []  # 所有图片文件列表 [(label, path), ...]
        self.dir1_files = []  # 目录1的文件列表
        self.dir2_files = []  # 目录2的文件列表
        self.paired_files = []  # 排序对比模式的配对文件列表
        self.visible_thumbnails = {}  # 当前可见的缩略图 {index: (frame, photo)}
        self.thumbnail_size = 48  # 缩略图尺寸
        self.thumbnails_per_row = 2  # 两列显示
        self.scroll_position = 0  # 当前滚动位置
        self.total_rows = 0  # 总行数
        self.thumbnail_row_height = 0
        
        # 配置列权重（两列布局，每列包含图片和文件名的垂直组合）
        self.thumbnail_frame.columnconfigure(0, weight=1)  # 目录1列
        self.thumbnail_frame.columnconfigure(1, weight=1)  # 目录2列

        # 右侧内容区域
        self.right_content = ttk.Frame(self.main_h_container)
        self.right_content.place(x=32, y=0, relwidth=1.0, relheight=1.0, width=-32)
        
        # Debug 预览区域（在 right_content 中，初始隐藏）
        self.debug_preview_frame = tk.Frame(self.right_content, bg='#ffffff')
        
        # 顶部：图片名称
        self.preview_name_label = tk.Label(self.debug_preview_frame, text="", bg='#e0e0e0', fg='#333333', 
                                           font=("Microsoft YaHei UI", 10, "bold"), anchor='w', padx=10, pady=5)
        self.preview_name_label.pack(fill='x')
        
        # 中间：图片预览区域
        self.preview_canvas = tk.Canvas(self.debug_preview_frame, bg='#f0f0f0', highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 绑定 ALT+鼠标滚轮缩放事件
        self.preview_canvas.bind('<MouseWheel>', self._on_preview_mousewheel)
        
        # 绑定鼠标拖动事件
        self.preview_canvas.bind('<ButtonPress-1>', self._on_preview_drag_start)
        self.preview_canvas.bind('<B1-Motion>', self._on_preview_drag_move)
        self.preview_canvas.bind('<ButtonRelease-1>', self._on_preview_drag_end)
        
        # 预览图拖动相关变量
        self.preview_drag_start_x = 0
        self.preview_drag_start_y = 0
        self.preview_offset_x = 0
        self.preview_offset_y = 0
        self.is_dragging = False
        
        # 底部：控制按钮
        button_frame = tk.Frame(self.debug_preview_frame, bg='#ffffff')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        points_btn = ttk.Radiobutton(button_frame, text="Points", variable=self.landmark_mode, 
                                     value='points', command=self._on_landmark_mode_changed)
        points_btn.pack(side='left', padx=5)
        
        lines_btn = ttk.Radiobutton(button_frame, text="Lines", variable=self.landmark_mode, 
                                    value='lines', command=self._on_landmark_mode_changed)
        lines_btn.pack(side='left', padx=5)
        
        mesh_btn = ttk.Radiobutton(button_frame, text="Mesh", variable=self.landmark_mode, 
                                   value='mesh', command=self._on_landmark_mode_changed)
        mesh_btn.pack(side='left', padx=5)
        
        # 正常内容区域（paned_window）
        self.normal_content_frame = ttk.Frame(self.right_content)
        self.normal_content_frame.pack(fill='both', expand=True)
        
        self.paned_window = ttk.PanedWindow(self.normal_content_frame, orient='vertical')
        self.paned_window.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.top_main_container = ttk.Frame(self.paned_window)
        self.paned_window.add(self.top_main_container, weight=3)
        
        self.notebook = ttk.Notebook(self.top_main_container)
        self.notebook.pack(fill='both', expand=True)
        
        self.extract_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.extract_frame, text='人脸提取 (Extract)')
        
        self.filter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.filter_frame, text='样本筛选 (Filter)')
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        self.setup_extract_tab()
        self.setup_filter_tab()
        
        self.bottom_container = ttk.Frame(self.paned_window)
        self.paned_window.add(self.bottom_container, weight=1)
        
        self.grip_frame = tk.Frame(self.bottom_container, bg='#e0e0e0', height=15, cursor='sb_v_double_arrow')
        self.grip_frame.pack(fill='x', side='top')
        
        grip_label = tk.Label(self.grip_frame, text="::::: 控制台输出 (Console) :::::", bg='#e0e0e0', fg='#555555', font=("Arial", 8), cursor='sb_v_double_arrow')
        grip_label.pack(expand=True)
        
        self.grip_frame.bind('<B1-Motion>', self.on_sash_drag)
        grip_label.bind('<B1-Motion>', self.on_sash_drag)
        
        self.paned_window.bind('<B1-Motion>', self.enforce_sash_limit, add='+')
        self.paned_window.bind('<ButtonRelease-1>', self.enforce_sash_limit, add='+')
        
        self.console_frame = ttk.Frame(self.bottom_container)
        self.console_frame.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(self.console_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.console = tk.Text(self.console_frame, height=15, font=("Consolas", 9), yscrollcommand=scrollbar.set)
        self.console.pack(fill='both', expand=True)
        scrollbar.config(command=self.console.yview)
        
        self.console.tag_config('error', foreground='red')
        self.console.tag_config('success', foreground='green')
        self.console.tag_config('info', foreground='black')
        self.console.tag_config('progress', foreground='blue')
        
        self.on_tab_changed(None)

    def on_sash_drag(self, event):
        try:
            y = self.paned_window.winfo_pointery() - self.paned_window.winfo_rooty()
            if y < 485:
                y = 485 
            self.paned_window.sashpos(0, y)
        except Exception:
            pass
            
    def enforce_sash_limit(self, event):
        try:
            if self.paned_window.sashpos(0) < 485:
                self.paned_window.sashpos(0, 485)
        except Exception:
            pass

    def on_tab_changed(self, event):
        selected_tab = self.notebook.select()
        if not selected_tab:
            return
        
        tab_index = self.notebook.index(selected_tab)
        
        for widget in self.sidebar_frame.winfo_children():
            widget.destroy()
            
        if tab_index == 0:
            self.update_sidebar(self.extract_sidebar_items, self.extract_pages)
        elif tab_index == 1:
            self.update_sidebar(self.filter_sidebar_items, self.filter_pages)
    
    def toggle_drawer(self, target_width=250):
        """切换抽屉展开/收起状态"""
        if self.drawer_expanded:
            # 收起抽屉
            self.drawer_target_width = 0
            self.drawer_expanded = False
        else:
            # 展开抽屉
            self.drawer_target_width = target_width
            self.drawer_expanded = True
        
        self.animate_drawer()
    
    def toggle_debug_mode(self):
        """切换 Debug 模式"""
        if self.drawer_expanded:
            # 退出 Debug 模式
            self.drawer_target_width = 0
            self.drawer_expanded = False
            # 显示正常内容，隐藏预览
            self.normal_content_frame.pack(fill='both', expand=True)
            self.debug_preview_frame.pack_forget()
            # Debug 按钮恢复默认背景
            if hasattr(self, 'debug_btn_canvas'):
                self.debug_btn_canvas.configure(bg='#f0f0f0')
        else:
            # 进入 Debug 模式
            self.drawer_target_width = 250
            self.drawer_expanded = True
            # 隐藏正常内容，显示预览
            self.normal_content_frame.pack_forget()
            self.debug_preview_frame.pack(fill='both', expand=True)
            # 取消所有侧边栏按钮的选中状态
            for widget in self.sidebar_frame.winfo_children():
                if isinstance(widget, tk.Canvas) and widget != self.debug_btn_canvas:
                    widget.configure(bg='#f0f0f0')
                    if hasattr(widget, 'is_active'):
                        widget.is_active = False
            # Debug 按钮设置为选中状态
            if hasattr(self, 'debug_btn_canvas'):
                self.debug_btn_canvas.configure(bg='#d0d0d0')
        
        self.animate_drawer()
    
    def animate_drawer(self):
        """抽屉动画"""
        if self.drawer_animation_id:
            self.after_cancel(self.drawer_animation_id)
            self.drawer_animation_id = None
        
        # 计算动画步长
        diff = self.drawer_target_width - self.drawer_width
        if abs(diff) < 1:
            self.drawer_width = self.drawer_target_width
            self.update_drawer_position()
            return
        
        # 平滑动画
        step = diff * 0.3
        if abs(step) < 1:
            step = 1 if diff > 0 else -1
        
        self.drawer_width += step
        self.update_drawer_position()
        
        # 继续动画
        self.drawer_animation_id = self.after(16, self.animate_drawer)  # ~60fps
    
    def update_drawer_position(self):
        """更新抽屉位置"""
        total_width = 32 + int(self.drawer_width)
        self.drawer_container.place(x=0, y=0, relheight=1.0, width=total_width)
        # 右侧内容始终显示，只是内部切换显示预览或正常内容
        self.right_content.place(x=total_width, y=0, relwidth=1.0, relheight=1.0, width=-total_width)
    
    def _on_thumbnail_canvas_configure(self, event):
        """缩略图 Canvas 大小改变时，重新渲染可见区域"""
        if self.total_rows > 0:
            self._render_visible_thumbnails()
            self._update_thumbnail_canvas_position()
    
    def _on_thumbnail_scroll(self, *args):
        """滚动条滚动事件"""
        if args[0] == 'moveto':
            # 滚动到指定位置
            position = float(args[1])
            self._scroll_to_position(position)
        elif args[0] == 'scroll':
            # 滚动指定步数
            delta = int(args[1])
            unit = args[2]
            if unit == 'units':
                # 按行滚动
                self._scroll_by_rows(delta)
            elif unit == 'pages':
                # 按页滚动
                self._scroll_by_pages(delta)
    
    def _on_thumbnail_mousewheel(self, event):
        """鼠标滚轮事件"""
        # Windows: event.delta 是 120 的倍数
        delta = -1 if event.delta > 0 else 1
        self._scroll_by_rows(delta)
    
    def _scroll_to_position(self, position):
        """滚动到指定位置（0.0 到 1.0）"""
        if self.total_rows == 0:
            return
        
        visible_rows = self._get_visible_rows()
        max_scroll = max(0, self.total_rows - visible_rows)
        
        # 计算滚动位置
        if position >= 0.95:  # 接近底部，确保显示最后的内容
            self.scroll_position = max_scroll
        elif position <= 0.01:  # 接近顶部
            self.scroll_position = 0
        else:
            self.scroll_position = int(position * max_scroll)
        
        self._render_visible_thumbnails()
        self._update_scrollbar()
    
    def _scroll_by_rows(self, delta):
        """按行滚动"""
        if self.total_rows == 0:
            return
        
        max_scroll = max(0, self.total_rows - self._get_visible_rows())
        self.scroll_position = max(0, min(max_scroll, self.scroll_position + delta))
        self._render_visible_thumbnails()
        self._update_scrollbar()
    
    def _scroll_by_pages(self, delta):
        """按页滚动"""
        visible_rows = self._get_visible_rows()
        self._scroll_by_rows(delta * visible_rows)
    
    def _get_visible_rows(self):
        """获取可见行数"""
        canvas_height = self.thumbnail_canvas.winfo_height()
        # 每行高度 = 缩略图(48) + 边框(4) + 文件名(约20) + 间距(4)
        row_height = self._get_row_height()
        return max(1, canvas_height // row_height)

    def _get_row_height(self):
        if self.thumbnail_row_height > 0:
            return self.thumbnail_row_height
        return self.thumbnail_size + 4 + 20 + 4

    def _update_thumbnail_canvas_position(self):
        row_height = self._get_row_height()
        total_height = self.total_rows * row_height
        canvas_height = self.thumbnail_canvas.winfo_height()
        if total_height < canvas_height:
            total_height = canvas_height
        self.thumbnail_canvas.configure(scrollregion=(0, 0, self.thumbnail_canvas.winfo_width(), total_height))
        self.thumbnail_canvas.coords(self.thumbnail_canvas_window, 0, 0)

    def _clamp_scroll_position(self):
        if self.total_rows == 0:
            self.scroll_position = 0
            return
        visible_rows = self._get_visible_rows()
        max_scroll = max(0, self.total_rows - visible_rows)
        if self.scroll_position > max_scroll:
            self.scroll_position = max_scroll
        if self.scroll_position < 0:
            self.scroll_position = 0
    
    def _update_scrollbar(self):
        """更新滚动条状态"""
        if self.total_rows == 0:
            self.thumbnail_scrollbar.set(0, 1)
            return
        
        self._clamp_scroll_position()
        visible_rows = self._get_visible_rows()
        if visible_rows >= self.total_rows:
            # 所有内容都可见
            self.thumbnail_scrollbar.set(0, 1)
        else:
            # 计算滚动条位置和大小
            first = self.scroll_position / self.total_rows
            last = min(1.0, (self.scroll_position + visible_rows) / self.total_rows)
            self.thumbnail_scrollbar.set(first, last)
    
    def on_debug_dir_changed(self):
        """目录输入框内容改变时，加载缩略图"""
        # 延迟加载，避免频繁刷新
        if hasattr(self, '_debug_dir_timer'):
            self.after_cancel(self._debug_dir_timer)
        self._debug_dir_timer = self.after(500, self.load_debug_thumbnails)
    
    def load_debug_thumbnails(self):
        """加载 Debug 目录的缩略图（扫描文件列表）"""
        if not self.turbojpeg_available:
            self.log("TurboJPEG 不可用，无法加载缩略图", "error")
            return
        
        # 清空现有数据
        self.dir1_files = []
        self.dir2_files = []
        self.visible_thumbnails = {}
        self.scroll_position = 0
        self.thumbnail_row_height = 0
        
        # 清空显示
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        
        dir1 = self.debug_dir1_var.get().strip()
        dir2 = self.debug_dir2_var.get().strip()
        
        # 扫描目录1
        if dir1 and Path(dir1).is_dir():
            dir_path = Path(dir1)
            files = [f for f in dir_path.iterdir() if f.is_file()]
            self.dir1_files = sorted(files, key=lambda x: x.name)
        
        # 扫描目录2
        if dir2 and Path(dir2).is_dir():
            dir_path = Path(dir2)
            files = [f for f in dir_path.iterdir() if f.is_file()]
            self.dir2_files = sorted(files, key=lambda x: x.name)
        
        if not self.dir1_files and not self.dir2_files:
            # 显示提示信息
            hint_label = tk.Label(self.thumbnail_frame, text="请输入有效的目录路径", 
                                 bg='#f5f5f5', fg='#999999', font=("Microsoft YaHei UI", 10))
            hint_label.grid(row=0, column=0, columnspan=2, sticky='nsew')
            self.total_rows = 0
            self.scroll_position = 0
            self._update_scrollbar()
            self._update_thumbnail_canvas_position()
            return
        
        # 根据是否排序对比，组织数据
        if self.sort_compare_var.get() and self.dir1_files and self.dir2_files:
            # 排序对比模式：按文件名匹配
            self._organize_sorted_pairs()
        else:
            # 普通模式：分别显示
            self._organize_separate_columns()
        
        # 渲染可见缩略图
        self._render_visible_thumbnails()
        self._update_scrollbar()
        
        total = len(self.dir1_files) + len(self.dir2_files)
        self.log(f"找到 {total} 个文件 (目录1: {len(self.dir1_files)}, 目录2: {len(self.dir2_files)})", "info")
    
    def _organize_sorted_pairs(self):
        """排序对比模式：按文件名匹配成对"""
        # 创建文件名到文件的映射
        dir1_map = {f.name: f for f in self.dir1_files}
        dir2_map = {f.name: f for f in self.dir2_files}
        
        # 获取所有文件名并排序
        all_names = sorted(set(dir1_map.keys()) | set(dir2_map.keys()))
        
        # 组织成对的数据：[(dir1_file or None, dir2_file or None), ...]
        self.paired_files = []
        for name in all_names:
            file1 = dir1_map.get(name)
            file2 = dir2_map.get(name)
            self.paired_files.append((file1, file2))
        
        self.total_rows = len(self.paired_files)
    
    def _organize_separate_columns(self):
        """普通模式：分别显示两列"""
        # 计算最大行数
        self.total_rows = max(len(self.dir1_files), len(self.dir2_files))
    
    def _render_visible_thumbnails(self):
        """渲染当前可见区域的缩略图（两列显示）"""
        if not self.dir1_files and not self.dir2_files:
            return
        
        # 保存当前选中的图片路径
        selected_img_path = None
        if self.selected_thumbnail_frame and self.selected_thumbnail_frame.winfo_exists():
            try:
                selected_img_path = self.selected_thumbnail_frame.img_path
            except:
                pass
        
        # 清空现有显示
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        self.visible_thumbnails = {}
        self.selected_thumbnail_frame = None  # 重置选中框架引用
        
        self._clamp_scroll_position()
        # 计算可见范围
        visible_rows = self._get_visible_rows() + 2  # 多加载 2 行作为缓冲
        start_index = self.scroll_position
        end_index = min(self.total_rows, start_index + visible_rows)
        
        # 确保滚动到底部时显示最后一行
        if self.scroll_position + visible_rows >= self.total_rows:
            end_index = self.total_rows
        
        # 根据模式渲染
        if self.sort_compare_var.get() and hasattr(self, 'paired_files'):
            # 排序对比模式
            self._render_paired_thumbnails(start_index, end_index)
        else:
            # 普通模式
            self._render_separate_thumbnails(start_index, end_index)
        
        if selected_img_path:
            self._restore_selection(selected_img_path)
        self._update_thumbnail_canvas_position()
    
    def _render_paired_thumbnails(self, start_index, end_index):
        """渲染排序对比模式的缩略图"""
        for i in range(start_index, end_index):
            if i >= len(self.paired_files):
                break
            
            file1, file2 = self.paired_files[i]
            row = i - start_index
            
            # 左列（目录1）
            if file1:
                self._create_thumbnail_widget(file1, row, 0)
            
            # 右列（目录2）
            if file2:
                self._create_thumbnail_widget(file2, row, 1)
    
    def _render_separate_thumbnails(self, start_index, end_index):
        """渲染普通模式的缩略图"""
        for i in range(start_index, end_index):
            row = i - start_index
            
            # 左列（目录1）
            if i < len(self.dir1_files):
                self._create_thumbnail_widget(self.dir1_files[i], row, 0)
            
            # 右列（目录2）
            if i < len(self.dir2_files):
                self._create_thumbnail_widget(self.dir2_files[i], row, 1)
    
    def _create_thumbnail_widget(self, img_file, row, col):
        """创建单个缩略图控件（垂直布局：图片在上，文件名在下）"""
        # 创建容器框架（垂直布局）
        container = tk.Frame(self.thumbnail_frame, bg='#f5f5f5')
        container.grid(row=row, column=col, padx=2, pady=2, sticky='n')
        
        # 上部：缩略图
        thumb_frame = tk.Frame(container, bg='#ffffff', 
                              relief='solid', bd=2, cursor='hand2',
                              width=self.thumbnail_size + 4,
                              height=self.thumbnail_size + 4,
                              highlightthickness=0)
        thumb_frame.pack(side='top')
        thumb_frame.pack_propagate(False)
        
        # 保存图片路径到框架，用于后续判断选中状态
        thumb_frame.img_path = img_file
        
        # 加载缩略图
        try:
            photo = debug_backend.load_thumbnail_turbojpeg(self.jpeg_decoder, img_file, self.thumbnail_size)
            if photo:
                img_label = tk.Label(thumb_frame, image=photo, bg='#ffffff', cursor='hand2')
                img_label.pack(expand=True)
                self.visible_thumbnails[f"{row}_{col}"] = (thumb_frame, photo)
                img_label.bind('<Button-1>', lambda e, frame=thumb_frame, path=img_file: self._on_thumbnail_click(path, frame))
            else:
                error_label = tk.Label(thumb_frame, text="✗", 
                                      bg='#ffffff', fg='#cc0000', 
                                      font=("Arial", 12))
                error_label.pack(expand=True)
        except Exception as e:
            error_label = tk.Label(thumb_frame, text="✗", 
                                  bg='#ffffff', fg='#cc0000', 
                                  font=("Arial", 12))
            error_label.pack(expand=True)
        
        # 下部：文件名
        filename_label = tk.Label(container, text=img_file.name, 
                                 bg='#f5f5f5', fg='#333333',
                                 font=("Arial", 8), height=2,
                                 cursor='hand2', wraplength=self.thumbnail_size + 4)
        filename_label.pack(side='top', pady=(2, 0))

        if self.thumbnail_row_height <= 0:
            container.update_idletasks()
            self.thumbnail_row_height = max(1, container.winfo_reqheight() + 4)
        
        # 绑定点击事件
        thumb_frame.bind('<Button-1>', lambda e, frame=thumb_frame, path=img_file: self._on_thumbnail_click(path, frame))
        container.bind('<Button-1>', lambda e, frame=thumb_frame, path=img_file: self._on_thumbnail_click(path, frame))
        filename_label.bind('<Button-1>', lambda e, frame=thumb_frame, path=img_file: self._on_thumbnail_click(path, frame))
    
    def _restore_selection(self, selected_img_path):
        """恢复选中状态"""
        # 遍历所有可见的缩略图，找到匹配的并设置选中状态
        for key, (thumb_frame, photo) in self.visible_thumbnails.items():
            try:
                if hasattr(thumb_frame, 'img_path') and thumb_frame.img_path == selected_img_path:
                    thumb_frame.configure(highlightthickness=2, highlightbackground='#ff0000')
                    self.selected_thumbnail_frame = thumb_frame
                    break
            except:
                pass

    
    def _on_thumbnail_click(self, img_path, thumb_frame):
        """点击缩略图时，在右侧显示大图并标记选中状态"""
        # 取消之前选中的缩略图的红框
        if self.selected_thumbnail_frame and self.selected_thumbnail_frame.winfo_exists():
            try:
                self.selected_thumbnail_frame.configure(highlightthickness=0, highlightbackground='#ffffff')
            except:
                pass
        
        # 设置当前缩略图为选中状态（红框）
        try:
            thumb_frame.configure(highlightthickness=2, highlightbackground='#ff0000')
            self.selected_thumbnail_frame = thumb_frame
        except:
            pass
        
        # 更新预览
        self.current_preview_image = img_path
        self.preview_name_label.config(text=img_path.name)
        self.preview_zoom_scale = 1.0  # 重置缩放比例
        self.preview_offset_x = 0  # 重置偏移
        self.preview_offset_y = 0
        self._update_preview()
    
    def _on_preview_drag_start(self, event):
        """开始拖动预览图"""
        self.preview_drag_start_x = event.x
        self.preview_drag_start_y = event.y
        self.is_dragging = True
        self.preview_canvas.config(cursor='fleur')  # 改变鼠标样式为移动
    
    def _on_preview_drag_move(self, event):
        """拖动预览图"""
        if not self.is_dragging:
            return
        
        # 计算拖动距离
        dx = event.x - self.preview_drag_start_x
        dy = event.y - self.preview_drag_start_y
        
        # 更新偏移量
        self.preview_offset_x += dx
        self.preview_offset_y += dy
        
        # 更新起始位置
        self.preview_drag_start_x = event.x
        self.preview_drag_start_y = event.y
        
        # 重新绘制预览图
        self._update_preview()
    
    def _on_preview_drag_end(self, event):
        """结束拖动预览图"""
        self.is_dragging = False
        self.preview_canvas.config(cursor='')  # 恢复默认鼠标样式
        self._update_preview()
    
    def _on_preview_mousewheel(self, event):
        """预览区域鼠标滚轮事件（ALT+滚轮缩放）"""
        # 检查是否按下 ALT 键
        if event.state & 0x20000:  # ALT key
            # 计算缩放增量
            if event.delta > 0:
                # 向上滚动，放大
                self.preview_zoom_scale *= 1.1
            else:
                # 向下滚动，缩小
                self.preview_zoom_scale /= 1.1
            
            # 限制缩放范围 0.1x ~ 5.0x
            self.preview_zoom_scale = max(0.1, min(5.0, self.preview_zoom_scale))
            
            # 更新预览
            self._update_preview()
            
            # 显示缩放比例
            self.log(f"预览缩放: {self.preview_zoom_scale:.1f}x", "info")
    
    def _on_landmark_mode_changed(self):
        """Landmark 模式改变时，重新绘制预览"""
        if self.current_preview_image:
            self._update_preview()
    
    def _update_preview(self):
        """更新右侧预览图片"""
        if not self.current_preview_image or not self.current_preview_image.exists():
            return
        
        try:
            import cv2
            import numpy as np
            
            # 读取图片
            img_bgr = cv2.imdecode(np.fromfile(str(self.current_preview_image), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                self.log(f"无法读取图片: {self.current_preview_image.name}", "error")
                return
            
            # 获取原始图片尺寸
            orig_h, orig_w = img_bgr.shape[:2]
            
            # 获取 landmarks（从 DFL 数据）
            landmarks = debug_backend.get_landmarks_from_image(self.current_preview_image)
            
            if landmarks is not None:
                # 根据模式绘制
                mode = self.landmark_mode.get()
                img_draw = img_bgr.copy()
                
                if mode == 'points':
                    # 绘制点和索引
                    debug_backend.draw_points(img_draw, landmarks)
                elif mode == 'lines':
                    # 绘制线条（使用 LandmarksProcessor）
                    try:
                        from facelib import LandmarksProcessor
                        LandmarksProcessor.draw_landmarks(img_draw, landmarks, transparent_mask=True, draw_circles=False)
                    except Exception as e:
                        self.log(f"绘制 landmarks 失败: {e}", "error")
                elif mode == 'mesh':
                    # 绘制网格（使用 LandmarksProcessor with circles）
                    try:
                        from facelib import LandmarksProcessor
                        LandmarksProcessor.draw_landmarks(img_draw, landmarks, transparent_mask=False, draw_circles=True)
                    except Exception as e:
                        self.log(f"绘制 landmarks 失败: {e}", "error")
                
                img_bgr = img_draw
            
            # 转换为 RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 获取当前预览区域大小
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # 自动调整UI大小（如果图片比预览区域大）
            resized = False
            if canvas_width > 1 and canvas_height > 1:
                # 如果图片比预览区域大，尝试扩大窗口
                if orig_w > canvas_width or orig_h > canvas_height:
                    resized = self._auto_resize_window(orig_w, orig_h)
                    if resized:
                        # 窗口调整后，需要重新获取canvas大小
                        self.update_idletasks()
                        canvas_width = self.preview_canvas.winfo_width()
                        canvas_height = self.preview_canvas.winfo_height()
                
                # 调整图片大小以适应预览区域，并应用缩放比例
                h, w = img_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                # 应用用户缩放比例
                scale *= self.preview_zoom_scale
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 转换为 PhotoImage
            h, w = img_rgb.shape[:2]
            ppm_header = f'P6 {w} {h} 255 '.encode()
            ppm_data = ppm_header + img_rgb.tobytes()
            photo = tk.PhotoImage(data=ppm_data, format='PPM')
            
            # 显示在 Canvas 中央（应用偏移量）
            self.preview_canvas.delete('all')
            center_x = canvas_width // 2 + self.preview_offset_x
            center_y = canvas_height // 2 + self.preview_offset_y
            self.preview_canvas.create_image(center_x, center_y, 
                                            image=photo, anchor='center')
            self.current_preview_photo = photo  # 保持引用
            
        except Exception as e:
            self.log(f"预览图片失败: {e}", "error")
            import traceback
            traceback.print_exc()
    
    def _auto_resize_window(self, img_width, img_height):
        """自动调整窗口宽度以适应图片（最大1024，禁止高度变化）"""
        try:
            # 获取当前窗口大小
            current_width = self.winfo_width()
            
            # 获取当前预览区域大小
            canvas_width = self.preview_canvas.winfo_width()
            
            # 只有当预览区域宽度不够显示图片时才调整（禁止高度变化）
            if canvas_width < img_width:
                # 计算需要的窗口宽度（考虑抽屉和边距）
                drawer_width = 250 + 32  # 抽屉宽度 + 侧边栏
                needed_width = img_width + drawer_width + 50  # 加50像素边距
                
                # 限制最大宽度为1024
                max_width = 1024
                needed_width = min(needed_width, max_width)
                
                # 只在需要扩大时调整宽度
                new_width = max(current_width, needed_width)
                
                # 如果需要调整
                if new_width != current_width:
                    # 只调整宽度，保持高度不变
                    current_height = self.winfo_height()
                    self.geometry(f"{new_width}x{current_height}")
                    self.log(f"窗口宽度已调整为 {new_width}", "info")
                    return True
            return False
        except Exception as e:
            return False
    
    def on_tab_changed_old(self, event):
        selected_tab = self.notebook.select()
        if not selected_tab:
            return
            
        tab_index = self.notebook.index(selected_tab)
        
        for widget in self.sidebar_frame.winfo_children():
            widget.destroy()
            
        if tab_index == 0:
            self.update_sidebar(self.extract_sidebar_items, self.extract_pages)
        elif tab_index == 1:
            self.update_sidebar(self.filter_sidebar_items, self.filter_pages)

    def update_sidebar(self, items, pages_dict):
        sidebar_tooltips = {
            "基础设置": "输入输出与提取参数",
            "指定人物": "指定人物提取与相关参数",
            "实时筛选": "提取时的实时筛选条件",
            "筛选设置": "样本筛选参数与输入输出",
            "Debug": "调试工具"
        }
        
        # 原有的侧边栏按钮
        for name, page_key in items:
            btn_canvas = tk.Canvas(self.sidebar_frame, width=32, height=100, bg='#f0f0f0', highlightthickness=0)
            btn_canvas.pack(side='top', fill='x', pady=1)
            
            cx, cy = 16, 50
            vertical_text = "\n".join(list(name))
            btn_canvas.create_text(cx, cy, text=vertical_text, font=("Microsoft YaHei UI", 9, "bold"), fill='#333333', anchor='center', justify='center')
            
            btn_canvas.bind("<Button-1>", lambda e, k=page_key, b=btn_canvas: self.switch_page(k, pages_dict, b))
            btn_canvas.bind("<Enter>", lambda e, b=btn_canvas: b.configure(bg='#e0e0e0'))
            btn_canvas.bind("<Leave>", lambda e, b=btn_canvas: self.check_btn_state(b))
            self.add_tooltip(btn_canvas, sidebar_tooltips.get(name, name))
            
            btn_canvas.is_active = False
        
        # 添加 Debug 按钮（在所有常规按钮之后，但不是最底部）
        debug_btn_canvas = tk.Canvas(self.sidebar_frame, width=32, height=80, bg='#f0f0f0', highlightthickness=0)
        debug_btn_canvas.pack(side='top', fill='x', pady=(10, 1))
        
        # Debug 图标和文字
        debug_btn_canvas.create_text(16, 40, text="D\nE\nB\nU\nG", 
                                     font=("Arial", 8, "bold"), fill='#333333', 
                                     anchor='center', justify='center')
        
        debug_btn_canvas.bind("<Button-1>", lambda e: self.toggle_debug_mode())
        debug_btn_canvas.bind("<Enter>", lambda e: debug_btn_canvas.configure(bg='#e0e0e0'))
        debug_btn_canvas.bind("<Leave>", lambda e: debug_btn_canvas.configure(bg='#f0f0f0'))
        self.add_tooltip(debug_btn_canvas, "打开/关闭 Debug 面板")
        
        # 保存 debug 按钮引用
        self.debug_btn_canvas = debug_btn_canvas
            
        if items:
            first_key = items[0][1]
            first_btn = None
            for child in self.sidebar_frame.winfo_children():
                if isinstance(child, tk.Canvas):
                    first_btn = child
                    break
            if first_btn is None:
                return
            self.switch_page(first_key, pages_dict, first_btn)

    def create_target_sidebar_toggle(self):
        wrapper = tk.Frame(self.sidebar_frame, bg='#f0f0f0')
        wrapper.pack(side='top', fill='x', pady=(2, 1))
        canvas = tk.Canvas(wrapper, width=32, height=100, bg='#f0f0f0', highlightthickness=0)
        canvas.pack(fill='x')
        self.sidebar_target_canvas = canvas
        self.refresh_target_sidebar()
        canvas.bind("<Button-1>", lambda e: self.on_toggle_target_sidebar())
        canvas.bind("<Enter>", lambda e: self.on_target_sidebar_hover(True))
        canvas.bind("<Leave>", lambda e: self.on_target_sidebar_hover(False))
        self.add_tooltip(canvas, "启用指定人物提取")

    def refresh_target_sidebar(self):
        canvas = getattr(self, "sidebar_target_canvas", None)
        if not canvas:
            return
        active = self.use_target_var.get()
        bg = '#cfe8cf' if active else '#f0f0f0'
        fg = '#1b7f3b' if active else '#333333'
        canvas.configure(bg=bg)
        canvas.delete("all")
        vertical_text = "\n".join(list("指定人物"))
        canvas.create_text(16, 50, text=vertical_text, font=("Microsoft YaHei UI", 9, "bold"), fill=fg, anchor='center', justify='center')

    def on_toggle_target_sidebar(self):
        self.use_target_var.set(not self.use_target_var.get())
        self.toggle_target_options()
        self.refresh_target_sidebar()

    def on_target_sidebar_hover(self, is_hover):
        canvas = getattr(self, "sidebar_target_canvas", None)
        if not canvas:
            return
        if is_hover:
            canvas.configure(bg='#e0e0e0')
        else:
            self.refresh_target_sidebar()

    def check_btn_state(self, btn):
        if not getattr(btn, 'is_active', False):
            btn.configure(bg='#f0f0f0')
        else:
            btn.configure(bg='#d0d0d0')

    def switch_page(self, page_key, pages_dict, btn_widget):
        # 如果在 Debug 模式，先退出
        if self.drawer_expanded:
            self.toggle_debug_mode()
        
        for page in pages_dict.values():
            page.pack_forget()
        
        if page_key in pages_dict:
            pages_dict[page_key].pack(fill='both', expand=True)
            
        for widget in self.sidebar_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.configure(bg='#f0f0f0')
                widget.is_active = False
        
        btn_widget.configure(bg='#d0d0d0')
        btn_widget.is_active = True

    def setup_extract_tab(self):
        self.extract_pages = {}
        self.extract_sidebar_items = [
            ("基础设置", "basic"),
            ("指定人物", "target"),
            ("实时筛选", "filter")
        ]
        
        basic_page = ttk.Frame(self.extract_frame)
        self.extract_pages["basic"] = basic_page
        self.create_extract_basic_page(basic_page)
        
        target_page = ttk.Frame(self.extract_frame)
        self.extract_pages["target"] = target_page
        self.create_extract_target_page(target_page)
        
        filter_page = ttk.Frame(self.extract_frame)
        self.extract_pages["filter"] = filter_page
        self.create_extract_filter_page(filter_page)

    def create_extract_basic_page(self, parent):
        settings_group = ttk.LabelFrame(parent, text=" 基础设置 (Basic Settings) ", padding=10)
        settings_group.pack(fill='x', pady=5)
        
        self.ex_input_var = tk.StringVar(value=self.get_conf("ex_input", ""))
        self.create_path_input(settings_group, "输入目录/视频:", self.ex_input_var, is_file="video_or_dir")
        
        self.ex_output_var = tk.StringVar(value=self.get_conf("ex_output", ""))
        self.create_path_input(settings_group, "输出目录:", self.ex_output_var, is_file=False)
        
        ttk.Separator(settings_group, orient='horizontal').pack(fill='x', pady=10)

        grid_frame = ttk.Frame(settings_group)
        grid_frame.pack(fill='x')
        
        ttk.Label(grid_frame, text="面部类型:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        face_types = ["whole_face", "full_face", "half_face", "head"]
        default_face_type = self.get_conf("face_type", "whole_face")
        if default_face_type not in face_types:
            default_face_type = "whole_face"
        self.face_type_var = tk.StringVar(value=default_face_type)
        ttk.OptionMenu(grid_frame, self.face_type_var, self.face_type_var.get(), *face_types).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Label(grid_frame, text="图片大小:").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.img_size_var = tk.IntVar(value=self.get_conf("img_size", 1024))
        size_combo = ttk.Combobox(grid_frame, textvariable=self.img_size_var, values=[1024, 768, 512, 256], width=8, state="readonly")
        size_combo.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(grid_frame, text="选择设备:").grid(row=0, column=6, padx=5, pady=5, sticky='w')
        self.gpu_idx_var = tk.StringVar(value=self.get_conf("gpu_idx_str", "GPU 0"))
        gpu_values = ["GPU 0", "GPU 1", "GPU 2", "GPU 3", "CPU"]
        gpu_combo = ttk.Combobox(grid_frame, textvariable=self.gpu_idx_var, values=gpu_values, width=8, state="readonly")
        gpu_combo.grid(row=0, column=7, padx=5, pady=5, sticky='ew')
        
        ttk.Label(grid_frame, text="JPEG 质量:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.jpeg_q_var = tk.IntVar(value=self.get_conf("jpeg_q", 100))
        ttk.Entry(grid_frame, textvariable=self.jpeg_q_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        loader_label = ttk.Label(grid_frame, text="加载线程:")
        loader_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.add_tooltip(loader_label, "加载图片/帧的线程数量")
        self.loader_threads_var = tk.IntVar(value=self.get_conf("loader_threads", min(4, multiprocessing.cpu_count())))
        ttk.Entry(grid_frame, textvariable=self.loader_threads_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        self.debug_var = tk.BooleanVar(value=self.get_conf("debug", False))
        ttk.Checkbutton(grid_frame, text="开启 Debug", variable=self.debug_var).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        
        save_no_face_cb = ttk.Checkbutton(grid_frame, text="保存未提取图片", variable=self.save_no_face_var)
        save_no_face_cb.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.add_tooltip(save_no_face_cb, "未检测到人脸时保存原图到 未提取 目录")

        saver_label = ttk.Label(grid_frame, text="保存线程:")
        saver_label.grid(row=2, column=2, padx=5, pady=5, sticky='w')
        self.add_tooltip(saver_label, "保存图片的线程数量")
        self.saver_threads_var = tk.IntVar(value=self.get_conf("saver_threads", max(1, multiprocessing.cpu_count() - 2)))
        ttk.Entry(grid_frame, textvariable=self.saver_threads_var, width=10).grid(row=2, column=3, padx=5, pady=5)

        model_group = ttk.LabelFrame(parent, text=" 模型切换 (Models) ", padding=10)
        model_group.pack(fill='x', pady=5)

        self.s3fd_model_var = tk.StringVar(value=self.get_conf("s3fd_model_path", "") or "默认")
        self.fan_model_var = tk.StringVar(value=self.get_conf("fan_model_path", "") or "默认")
        self.rec_model_var = tk.StringVar(value=self.get_conf("rec_model_path", "") or "默认")
        self.model_candidates = self.get_model_candidates()

        self.create_model_input(model_group, "检测模型(S3FD):", self.s3fd_model_var, self.model_candidates.get("s3fd", []))
        self.create_model_input(model_group, "对齐模型(FAN):", self.fan_model_var, self.model_candidates.get("fan", []))
        self.create_model_input(model_group, "识别模型(Insight):", self.rec_model_var, self.model_candidates.get("rec", []))

        video_group = ttk.LabelFrame(parent, text=" 视频选项 (Video) ", padding=10)
        video_group.pack(fill='x', pady=5)
        
        v_grid = ttk.Frame(video_group)
        v_grid.pack(fill='x')
        
        skip_start_label = ttk.Label(v_grid, text="跳过片头(s):")
        skip_start_label.grid(row=0, column=0, padx=5, pady=5)
        self.add_tooltip(skip_start_label, "跳过视频开始的秒数")
        self.skip_start_var = tk.DoubleVar(value=self.get_conf("skip_start", 0.0))
        skip_start_entry = ttk.Entry(v_grid, textvariable=self.skip_start_var, width=8)
        skip_start_entry.grid(row=0, column=1, padx=5, pady=5)
        
        skip_end_label = ttk.Label(v_grid, text="跳过片尾(s):")
        skip_end_label.grid(row=0, column=2, padx=5, pady=5)
        self.add_tooltip(skip_end_label, "跳过视频结尾的秒数")
        self.skip_end_var = tk.DoubleVar(value=self.get_conf("skip_end", 0.0))
        skip_end_entry = ttk.Entry(v_grid, textvariable=self.skip_end_var, width=8)
        skip_end_entry.grid(row=0, column=3, padx=5, pady=5)
        
        step_label = ttk.Label(v_grid, text="抽帧步长:")
        step_label.grid(row=0, column=4, padx=5, pady=5)
        self.add_tooltip(step_label, "步长为 N 时，每隔 N 帧提取 1 帧 ，0 表示每帧都提取")
        self.step_var = tk.IntVar(value=self.get_conf("step", 1))
        step_entry = ttk.Entry(v_grid, textvariable=self.step_var, width=8)
        step_entry.grid(row=0, column=5, padx=5, pady=5)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', pady=10)
        
        start_btn = tk.Button(btn_frame, text="开始提取 (Start)", command=self.run_extract, bg='green', fg='white', font=("Arial", 10, "bold"))
        start_btn.pack(side='left', fill='x', expand=True, padx=5)
        self.add_tooltip(start_btn, "开始执行当前输入目录/视频的提取")

        stop_btn = tk.Button(btn_frame, text="停止 (Stop)", command=self.stop_process, bg='red', fg='white', font=("Arial", 10, "bold"))
        stop_btn.pack(side='left', fill='x', expand=True, padx=5)
        self.add_tooltip(stop_btn, "停止当前正在运行的提取任务")

        batch_btn = tk.Button(btn_frame, text="批量提取 (Batch)", command=self.run_batch_extract, bg='#007acc', fg='white', font=("Arial", 10, "bold"))
        batch_btn.pack(side='left', fill='x', expand=True, padx=5)
        self.add_tooltip(batch_btn, "对输入目录下所有视频执行批量提取")

        reset_btn = tk.Button(btn_frame, text="恢复默认 (Reset)", command=self.reset_to_defaults, bg='#666666', fg='white', font=("Arial", 10, "bold"))
        reset_btn.pack(side='left', fill='x', expand=True, padx=5)
        self.add_tooltip(reset_btn, "恢复为当前默认设置")

    def create_extract_target_page(self, parent):
        target_group = ttk.LabelFrame(parent, text=" 指定人物设置 (Target) ", padding=10)
        target_group.pack(fill='x', pady=5)
        
        self.use_target_var = tk.BooleanVar(value=self.get_conf("use_target", False))
        ttk.Checkbutton(target_group, text="启用指定人物提取", variable=self.use_target_var, command=self.toggle_target_options).pack(anchor='w')
        
        self.target_opts_frame = ttk.Frame(target_group)
        self.target_opts_frame.pack(fill='x', pady=5)
        
        self.ref_dir_var = tk.StringVar(value=self.get_conf("ref_dir", ""))
        self.create_path_input(self.target_opts_frame, "参考人脸目录:", self.ref_dir_var, is_file=False)
        
        target_grid = ttk.Frame(self.target_opts_frame)
        target_grid.pack(fill='x')
        
        ttk.Label(target_grid, text="相似度阈值:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.sim_thresh_var = tk.DoubleVar(value=self.get_conf("sim_thresh", 0.4))
        sim_entry = ttk.Entry(target_grid, textvariable=self.sim_thresh_var, width=10)
        sim_entry.grid(row=0, column=1, padx=5, pady=5)
        self.add_tooltip(sim_entry, "人脸相似度阈值，越大越严格")
        
        self.multi_target_var = tk.BooleanVar(value=self.get_conf("multi_target", False))
        multi_target_cb = ttk.Checkbutton(target_grid, text="多人物", variable=self.multi_target_var)
        multi_target_cb.grid(row=0, column=2, padx=15, pady=5)
        self.add_tooltip(multi_target_cb, "参考目录内按子目录区分多个人物")
        
        self.keep_non_target_var = tk.BooleanVar(value=self.get_conf("keep_non_target", False))
        keep_non_target_cb = ttk.Checkbutton(target_grid, text="保存非目标人物", variable=self.keep_non_target_var)
        keep_non_target_cb.grid(row=0, column=3, padx=15, pady=5)
        self.add_tooltip(keep_non_target_cb, "将非目标人物保存到单独目录")
        
        self.reduce_non_target_var = tk.BooleanVar(value=self.get_conf("reduce_non_target", False))
        reduce_non_target_cb = ttk.Checkbutton(target_grid, text="减少非目标保存", variable=self.reduce_non_target_var)
        reduce_non_target_cb.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.add_tooltip(reduce_non_target_cb, "减少非目标样本保存数量")
        
        ttk.Label(target_grid, text="保留间隔:").grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.non_target_keep_interval_var = tk.IntVar(value=self.get_conf("non_target_keep_interval", 5))
        ttk.Entry(target_grid, textvariable=self.non_target_keep_interval_var, width=8).grid(row=1, column=2, padx=5, pady=5, sticky='w')
        
        self.auto_augment_var = tk.BooleanVar(value=self.get_conf("auto_augment", False))
        auto_augment_cb = ttk.Checkbutton(target_grid, text="自动补充样本(未支持)", variable=self.auto_augment_var, state='disabled')
        auto_augment_cb.grid(row=0, column=4, padx=15, pady=5)
        self.add_tooltip(auto_augment_cb, "版本尚未支持该功能")
        
        self.toggle_target_options()

    def create_extract_filter_page(self, parent):
        adv_group = ttk.LabelFrame(parent, text=" 实时筛选 (Filter) ", padding=10)
        adv_group.pack(fill='x', pady=5)
        
        self.rt_blur_var = tk.BooleanVar(value=self.get_conf("rt_blur", False))
        ttk.Checkbutton(adv_group, text="开启模糊分类", variable=self.rt_blur_var).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Label(adv_group, text="阈值(低/高):").grid(row=0, column=1, padx=5)
        self.rt_blur_low_var = tk.DoubleVar(value=self.get_conf("rt_blur_low", 10.0))
        ttk.Entry(adv_group, textvariable=self.rt_blur_low_var, width=5).grid(row=0, column=2, padx=2)
        self.rt_blur_high_var = tk.DoubleVar(value=self.get_conf("rt_blur_high", 20.0))
        ttk.Entry(adv_group, textvariable=self.rt_blur_high_var, width=5).grid(row=0, column=3, padx=2)
        
        self.rt_mouth_var = tk.BooleanVar(value=self.get_conf("rt_mouth", False))
        ttk.Checkbutton(adv_group, text="挑选张嘴图片", variable=self.rt_mouth_var).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Label(adv_group, text="张嘴阈值:").grid(row=1, column=1, padx=5)
        self.rt_mouth_thresh_var = tk.DoubleVar(value=self.get_conf("rt_mouth_thresh", 15.0))
        ttk.Entry(adv_group, textvariable=self.rt_mouth_thresh_var, width=5).grid(row=1, column=2, padx=2)
        
        self.rt_pose_var = tk.BooleanVar(value=self.get_conf("rt_pose", False))
        rt_pose_cb = ttk.Checkbutton(adv_group, text="挑选角度图片", variable=self.rt_pose_var, command=self.on_rt_pose_toggle)
        rt_pose_cb.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Label(adv_group, text="阈值(Pitch/Yaw):").grid(row=2, column=1, padx=5)
        self.rt_pitch_var = tk.DoubleVar(value=self.get_conf("rt_pitch", 10.0))
        ttk.Entry(adv_group, textvariable=self.rt_pitch_var, width=5).grid(row=2, column=2, padx=2)
        self.rt_yaw_var = tk.DoubleVar(value=self.get_conf("rt_yaw", 45.0))
        ttk.Entry(adv_group, textvariable=self.rt_yaw_var, width=5).grid(row=2, column=3, padx=2)
        
        self.rt_pose_dir_var = tk.StringVar(value=self.get_conf("rt_pose_dir", "抬头"))
        dir_frame = ttk.Frame(adv_group)
        dir_frame.grid(row=3, column=0, columnspan=4, sticky='w', padx=5)
        ttk.Label(dir_frame, text="目标角度:").pack(side='left')
        self.rt_pose_dir_buttons = []
        for d in ["抬头", "低头", "向左", "向右"]:
            rb = ttk.Radiobutton(dir_frame, text=d, variable=self.rt_pose_dir_var, value=d)
            rb.pack(side='left', padx=5)
            self.rt_pose_dir_buttons.append(rb)
        self.on_rt_pose_toggle()

    def setup_filter_tab(self):
        self.filter_pages = {}
        self.filter_sidebar_items = [
            ("筛选设置", "filter_settings")
        ]
        
        filter_page = ttk.Frame(self.filter_frame)
        self.filter_pages["filter_settings"] = filter_page
        self.create_filter_settings_page(filter_page)
        
    def create_filter_settings_page(self, parent):
        io_group = ttk.LabelFrame(parent, text=" 筛选 I/O (Filter I/O) ", padding=10)
        io_group.pack(fill='x', pady=5)
        
        self.filter_input_var = tk.StringVar(value=self.get_conf("filter_input", ""))
        self.create_path_input(io_group, "输入目录:", self.filter_input_var, is_file=False)
        
        self.filter_output_var = tk.StringVar(value=self.get_conf("filter_output", ""))
        self.create_path_input(io_group, "输出目录:", self.filter_output_var, is_file=False)
        
        self.filter_ref_dir_var = tk.StringVar(value=self.get_conf("filter_ref_dir", ""))
        self.filter_sim_thresh_var = tk.DoubleVar(value=self.get_conf("filter_sim_thresh", 0.4))
        self.filter_multi_target_var = tk.BooleanVar(value=self.get_conf("filter_multi_target", False))
        self.filter_target_pick_var = tk.BooleanVar(value=self.get_conf("filter_target_pick", True))
        
        mode_group = ttk.LabelFrame(parent, text=" 筛选模式 (Mode) ", padding=10)
        mode_group.pack(fill='x', pady=5)
        
        self.filter_mode_var = tk.StringVar(value=self.get_conf("filter_mode", "pose"))
        modes = [("角度筛选 (Pose)", "pose"), ("张嘴筛选 (Mouth)", "mouth"), ("模糊分类 (Blur)", "blur"), ("指定人物筛选 (Target)", "target")]
        
        for text, val in modes:
            ttk.Radiobutton(mode_group, text=text, variable=self.filter_mode_var, value=val, command=self.update_filter_ui).pack(anchor='w', pady=2)
            
        self.filter_settings_frame = ttk.Frame(mode_group)
        self.filter_settings_frame.pack(fill='x', pady=5, padx=20)
        
        self.pose_frame = ttk.Frame(self.filter_settings_frame)
        self.pose_var = tk.StringVar(value=self.get_conf("pose_single", "")) # Changed to StringVar
        # "仅保留单人" seems broken/unused in original logic (it gets destroyed immediately), so we skip it or fix it.
        # Since backend doesn't support it, we skip it.
        
        ttk.Label(self.pose_frame, text="Pitch 阈值:").pack(side='left')
        self.pitch_var = tk.DoubleVar(value=self.get_conf("pitch_thresh", 15.0))
        ttk.Entry(self.pose_frame, textvariable=self.pitch_var, width=5).pack(side='left', padx=5)
        ttk.Label(self.pose_frame, text="Yaw 阈值:").pack(side='left')
        self.yaw_var = tk.DoubleVar(value=self.get_conf("yaw_thresh", 20.0))
        ttk.Entry(self.pose_frame, textvariable=self.yaw_var, width=5).pack(side='left', padx=5)
        
        self.mouth_frame = ttk.Frame(self.filter_settings_frame)
        ttk.Label(self.mouth_frame, text="张嘴阈值:").pack(side='left')
        self.mouth_thresh_var = tk.DoubleVar(value=self.get_conf("mouth_thresh", 20.0))
        ttk.Entry(self.mouth_frame, textvariable=self.mouth_thresh_var, width=5).pack(side='left', padx=5)
        
        self.blur_frame = ttk.Frame(self.filter_settings_frame)
        ttk.Label(self.blur_frame, text="清晰度下限:").pack(side='left')
        self.blur_low_var = tk.DoubleVar(value=self.get_conf("blur_low", 100.0))
        ttk.Entry(self.blur_frame, textvariable=self.blur_low_var, width=5).pack(side='left', padx=5)
        ttk.Label(self.blur_frame, text="上限:").pack(side='left')
        self.blur_high_var = tk.DoubleVar(value=self.get_conf("blur_high", 500.0))
        ttk.Entry(self.blur_frame, textvariable=self.blur_high_var, width=5).pack(side='left', padx=5)
        
        self.update_filter_ui()
        
        filter_btn = tk.Button(parent, text="开始筛选 (Start Filter)", command=self.run_filter, bg='blue', fg='white', font=("Arial", 10, "bold"))
        filter_btn.pack(fill='x', pady=20)
        self.add_tooltip(filter_btn, "对输入目录中的样本执行筛选")

    def get_model_candidates(self):
        root_dir = Path(__file__).parent.parent.parent
        search_dirs = [
            root_dir / "assets" / "models",
            root_dir / "assets" / "models_exported",
            root_dir / "models_exported",
            root_dir / "TF_Extract.dist" / "models",
            root_dir / "main.dist" / "models",
            root_dir / "_libs" / "models"
        ]
        
        self.model_name_map = {"默认": ""}
        
        s3fd_names = []
        fan_names = []
        rec_names = []
        
        for d in search_dirs:
            if not d.exists():
                continue
            try:
                for p in d.rglob("*.onnx"):
                    path_str = str(p)
                    name = p.name
                    
                    # Simple duplicate handling: if name exists but path differs, keep first or overwrite?
                    # Let's try to make name unique if needed, but for now simple overwrite or keep is fine.
                    # Or check if path is already mapped (reverse lookup not efficient but OK here).
                    
                    # If we encounter same name with different path, maybe prepend parent dir?
                    if name in self.model_name_map and self.model_name_map[name] != path_str:
                         # Try name with parent
                         name = f"{p.parent.name}/{p.name}"
                    
                    self.model_name_map[name] = path_str
                    
                    lower_name = p.name.lower()
                    if "s3fd" in lower_name or "det_10g" in lower_name or "scrfd" in lower_name:
                        s3fd_names.append(name)
                    elif "fan" in lower_name or "2d106det" in lower_name or "1k3d68" in lower_name:
                        fan_names.append(name)
                    else:
                        rec_names.append(name)
            except Exception:
                continue
                
        def uniq(items):
            seen = set()
            out = []
            for it in items:
                if it in seen:
                    continue
                seen.add(it)
                out.append(it)
            return out
            
        return {
            "s3fd": uniq(sorted(s3fd_names)),
            "fan": uniq(sorted(fan_names)),
            "rec": uniq(sorted(rec_names))
        }

    def create_model_input(self, parent, label_text, real_var, values):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        ttk.Label(frame, text=label_text, width=15).pack(side='left')
        
        display_var = tk.StringVar()
        
        def on_real_change(*args):
            val = real_var.get()
            if not val or val == "默认":
                display_var.set("默认")
                return
            
            # Find name for this path
            found_name = None
            for name, path in self.model_name_map.items():
                if str(path) == str(val):
                    found_name = name
                    break
            
            if found_name:
                display_var.set(found_name)
            else:
                # Not in map, just show filename
                try:
                    p = Path(val)
                    name = p.name
                    display_var.set(name)
                    # Add to map so we can select it back if needed? 
                    # But wait, if user selects it back from combobox, we need map entry.
                    self.model_name_map[name] = val
                except:
                    display_var.set(val)

        # Trace changes to real_var (config load or browse)
        real_var.trace("w", on_real_change)
        
        options = ["默认"] + list(values)
        combo = ttk.Combobox(frame, textvariable=display_var, values=options, state="normal")
        combo.pack(side='left', fill='x', expand=True, padx=5)
        
        # 启用拖放功能
        if TKDND_AVAILABLE:
            combo.drop_target_register(DND_FILES)
            
            def on_drop(event):
                data = event.data
                if data.startswith('{') and data.endswith('}'):
                    data = data[1:-1]
                data = data.strip('"').strip("'")
                
                path = Path(data)
                if path.exists() and path.is_file() and path.suffix.lower() == '.onnx':
                    real_var.set(str(path).replace('/', '\\'))
            
            combo.dnd_bind('<<Drop>>', on_drop)
        
        def on_display_change(event=None):
            name = display_var.get()
            if name in self.model_name_map:
                path = self.model_name_map[name]
                if real_var.get() != path:
                    real_var.set(path)
            else:
                # User typed something that is not in map?
                # Maybe they typed a path?
                pass

        combo.bind("<<ComboboxSelected>>", on_display_change)
        # Also bind Return to handle manual typing?
        combo.bind("<Return>", on_display_change)
        combo.bind("<FocusOut>", on_display_change)
        
        # Trigger initial sync
        on_real_change()

        btn = ttk.Button(frame, text="...", width=3, command=lambda: self.browse_model_file(real_var))
        btn.pack(side='left')
        self.add_tooltip(btn, "选择 ONNX 模型文件")

    def browse_model_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("ONNX 模型", "*.onnx"), ("所有文件", "*.*")])
        if path:
            path = path.replace('/', '\\')
            var.set(path)

    def create_path_input(self, parent, label_text, var, is_file=False):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        ttk.Label(frame, text=label_text, width=15).pack(side='left')
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # 启用拖放功能
        if TKDND_AVAILABLE:
            entry.drop_target_register(DND_FILES)
            
            def on_drop(event):
                data = event.data
                # 处理Windows路径格式
                if data.startswith('{') and data.endswith('}'):
                    data = data[1:-1]
                data = data.strip('"').strip("'")
                
                path = Path(data)
                if not path.exists():
                    return
                
                # 根据is_file参数验证类型
                if is_file == "video_or_dir":
                    var.set(str(path).replace('/', '\\'))
                elif is_file:
                    if path.is_file():
                        var.set(str(path).replace('/', '\\'))
                else:
                    if path.is_dir():
                        var.set(str(path).replace('/', '\\'))
                    elif path.is_file():
                        var.set(str(path.parent).replace('/', '\\'))
            
            entry.dnd_bind('<<Drop>>', on_drop)
        
        if is_file == "video_or_dir":
            btn_file = ttk.Button(frame, text="文件", width=4, command=lambda: self.browse_path(var, True))
            btn_file.pack(side='left', padx=(0, 4))
            self.add_tooltip(btn_file, "选择视频文件")
            btn_dir = ttk.Button(frame, text="目录", width=4, command=lambda: self.browse_path(var, False))
            btn_dir.pack(side='left')
            self.add_tooltip(btn_dir, "选择目录")
        else:
            cmd = lambda: self.browse_path(var, is_file)
            browse_btn = ttk.Button(frame, text="...", width=3, command=cmd)
            browse_btn.pack(side='left')
            self.add_tooltip(browse_btn, f"选择{label_text.replace(':', '').strip()}")

    def browse_path(self, var, is_file):
        if is_file == "video_or_dir":
            path = self.choose_file_or_dir(var.get())
        elif is_file:
            video_types = ("*.mp4;*.avi;*.mkv;*.mov;*.flv;*.wmv;*.webm;*.ts;*.m4v")
            path = filedialog.askopenfilename(filetypes=[("视频文件", video_types), ("所有文件", "*.*")])
        else:
            path = filedialog.askdirectory()
            
        if path:
            var.set(path.replace('/', '\\'))

    def choose_file_or_dir(self, initial_value=""):
        dialog = tk.Toplevel(self)
        dialog.title("选择输入目录或视频")
        dialog.geometry("720x420")
        dialog.transient(self)
        dialog.grab_set()
        result = {"path": None}
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".ts", ".m4v"}
        def resolve_start():
            p = Path(initial_value) if initial_value else None
            if p and p.exists():
                return p if p.is_dir() else p.parent
            return Path.cwd()
        current_dir = resolve_start()
        path_var = tk.StringVar(value=str(current_dir))
        top = ttk.Frame(dialog)
        top.pack(fill='x', padx=8, pady=6)
        ttk.Entry(top, textvariable=path_var).pack(side='left', fill='x', expand=True, padx=(0, 6))
        def go_up():
            nonlocal current_dir
            parent = current_dir.parent
            if parent and parent.exists():
                current_dir = parent
                path_var.set(str(current_dir))
                load_dir()
        ttk.Button(top, text="上级", width=6, command=go_up).pack(side='left')
        tree = ttk.Treeview(dialog, columns=("type",), show="tree headings")
        tree.heading("#0", text="名称")
        tree.heading("type", text="类型")
        tree.column("type", width=80, anchor="center")
        tree.pack(fill='both', expand=True, padx=8, pady=(0, 6))
        def load_dir():
            tree.delete(*tree.get_children())
            try:
                entries = list(os.scandir(current_dir))
            except Exception:
                return
            dirs = [e for e in entries if e.is_dir()]
            files = [e for e in entries if e.is_file() and e.name.lower().endswith(tuple(video_exts))]
            for e in sorted(dirs, key=lambda x: x.name.lower()):
                tree.insert("", "end", text=e.name, values=("目录",), tags=("dir",))
            for e in sorted(files, key=lambda x: x.name.lower()):
                tree.insert("", "end", text=e.name, values=("文件",), tags=("file",))
        def on_open(event=None):
            nonlocal current_dir
            sel = tree.selection()
            if not sel:
                return
            name = tree.item(sel[0], "text")
            path = current_dir / name
            if path.is_dir():
                current_dir = path
                path_var.set(str(current_dir))
                load_dir()
            elif path.is_file():
                result["path"] = str(path)
                dialog.destroy()
        tree.bind("<Double-1>", on_open)
        btns = ttk.Frame(dialog)
        btns.pack(fill='x', padx=8, pady=6)
        def choose_selected():
            sel = tree.selection()
            if not sel:
                return
            name = tree.item(sel[0], "text")
            path = current_dir / name
            if path.is_dir() or path.is_file():
                result["path"] = str(path)
                dialog.destroy()
        ttk.Button(btns, text="选择", command=choose_selected).pack(side='right', padx=(6, 0))
        ttk.Button(btns, text="取消", command=dialog.destroy).pack(side='right')
        load_dir()
        dialog.wait_window(dialog)
        return result["path"]

    def on_rt_pose_toggle(self):
        enabled = self.rt_pose_var.get()
        state = 'normal' if enabled else 'disabled'
        for rb in getattr(self, 'rt_pose_dir_buttons', []):
            rb.configure(state=state)
        if not enabled:
            self.rt_pose_dir_var.set("")
        else:
            if not self.rt_pose_dir_var.get():
                self.rt_pose_dir_var.set("抬头")

    def toggle_target_options(self):
        state = 'normal' if self.use_target_var.get() else 'disabled'
        for child in self.target_opts_frame.winfo_children():
            self.set_state_recursive(child, state)
            
    def set_state_recursive(self, widget, state):
        try:
            widget.configure(state=state)
        except:
            pass
        for child in widget.winfo_children():
            self.set_state_recursive(child, state)

    def update_filter_ui(self):
        for widget in self.filter_settings_frame.winfo_children():
            widget.destroy()
            
        mode = self.filter_mode_var.get()
        frame = ttk.LabelFrame(self.filter_settings_frame, text=f" {mode.title()} 设置 ", padding=10)
        frame.pack(fill='x')
        
        if mode == 'pose':
            p_frame = ttk.Frame(frame)
            p_frame.pack(fill='x', pady=5)
            ttk.Radiobutton(p_frame, text='抬头', variable=self.pose_var, value='抬头').pack(side='left', padx=5)
            ttk.Radiobutton(p_frame, text='低头', variable=self.pose_var, value='低头').pack(side='left', padx=5)
            ttk.Radiobutton(p_frame, text='向左', variable=self.pose_var, value='向左').pack(side='left', padx=5)
            ttk.Radiobutton(p_frame, text='向右', variable=self.pose_var, value='向右').pack(side='left', padx=5)
            
            t_frame = ttk.Frame(frame)
            t_frame.pack(fill='x', pady=5)
            ttk.Label(t_frame, text="Pitch 阈值:").pack(side='left')
            ttk.Entry(t_frame, textvariable=self.pitch_var, width=8).pack(side='left', padx=5)
            
            ttk.Label(t_frame, text="Yaw 阈值:").pack(side='left', padx=(15, 0))
            ttk.Entry(t_frame, textvariable=self.yaw_var, width=8).pack(side='left', padx=5)
            
        elif mode == 'mouth':
            ttk.Label(frame, text="张嘴阈值:").pack(side='left')
            ttk.Entry(frame, textvariable=self.mouth_thresh_var, width=8).pack(side='left', padx=5)
            ttk.Label(frame, text="(越大越张)").pack(side='left')
            
        elif mode == 'blur':
            ttk.Label(frame, text="模糊下限:").pack(side='left')
            ttk.Entry(frame, textvariable=self.blur_low_var, width=8).pack(side='left', padx=5)
            
            ttk.Label(frame, text="模糊上限:").pack(side='left', padx=(15, 0))
            ttk.Entry(frame, textvariable=self.blur_high_var, width=8).pack(side='left', padx=5)
        
        elif mode == 'target':
            self.create_path_input(frame, "参考人脸目录:", self.filter_ref_dir_var, is_file=False)
            
            target_grid = ttk.Frame(frame)
            target_grid.pack(fill='x', pady=5)
            
            ttk.Label(target_grid, text="相似度阈值:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            sim_entry = ttk.Entry(target_grid, textvariable=self.filter_sim_thresh_var, width=10)
            sim_entry.grid(row=0, column=1, padx=5, pady=5)
            self.add_tooltip(sim_entry, "人脸相似度阈值，越大越严格")
            
            multi_target_cb = ttk.Checkbutton(target_grid, text="多人物", variable=self.filter_multi_target_var)
            multi_target_cb.grid(row=0, column=2, padx=15, pady=5)
            self.add_tooltip(multi_target_cb, "参考目录内按子目录区分多个人物")
            
            pick_target_cb = ttk.Checkbutton(frame, text="挑出目标人物", variable=self.filter_target_pick_var)
            pick_target_cb.pack(anchor='w', pady=5)
            self.add_tooltip(pick_target_cb, "不勾选时输出非目标人物")
            

    def log(self, message, level='info'):
        tags = (level,)
        if self.log_file:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                if level != 'progress':
                    self.log_file.write(f"[{ts}] [{level.upper()}] {message}\n")
                    self.log_file.flush()
            except Exception:
                pass
        if level == 'progress':
            if self.last_msg_was_progress:
                self.console.delete("end-2l", "end-1l")
            self.console.insert('end', message + '\n', tags)
            self.last_msg_was_progress = True
        else:
            self.console.insert('end', message + '\n', tags)
            self.last_msg_was_progress = False
        self.console.see('end')

    def check_queue(self):
        while not self.log_queue.empty():
            try:
                msg, level = self.log_queue.get_nowait()
                self.log(msg, level)
            except queue.Empty:
                break
        
        # Check process status
        if self.process and not self.process.is_alive():
             self.log(f"Process finished with exit code: {self.process.exitcode}", 
                      'success' if self.process.exitcode == 0 else 'error')
             self.process = None
             
        self.after(50, self.check_queue)

    def run_worker(self, target_func, args):
        if self.process and self.process.is_alive():
            messagebox.showwarning("警告", "已有任务正在运行")
            return
            
        self.process = multiprocessing.Process(
            target=target_func,
            args=(args, self.log_queue)
        )
        self.process.start()

    def stop_process(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.log("Stopping process...", 'error')

    def build_extract_args(self, input_path=None, output_path=None):
        in_path = input_path if input_path else self.ex_input_var.get()
        if not in_path:
            return None, "请输入输入路径"
        
        out_path = output_path if output_path else self.ex_output_var.get()
        if not out_path:
            return None, "请输入输出路径"
            
        gpu_str = self.gpu_idx_var.get()
        if "CPU" in gpu_str:
            gpu_val = -1
        else:
            try:
                gpu_val = int(gpu_str.split()[-1])
            except:
                gpu_val = 0
                
        args = {
            "input_dir": in_path,
            "output_dir": out_path,
            "face_type": self.face_type_var.get(),
            "image_size": self.img_size_var.get(),
            "gpu_idx": gpu_val,
            "jpeg_quality": self.jpeg_q_var.get(),
            "loader_threads": self.loader_threads_var.get(),
            "saver_threads": self.saver_threads_var.get(),
            "debug": self.debug_var.get(),
            "save_no_face": self.save_no_face_var.get(),
            "s3fd_model_path": "" if self.s3fd_model_var.get() in ["", "默认"] else self.s3fd_model_var.get(),
            "fan_model_path": "" if self.fan_model_var.get() in ["", "默认"] else self.fan_model_var.get(),
            "rec_model_path": "" if self.rec_model_var.get() in ["", "默认"] else self.rec_model_var.get(),
            "skip_start": self.skip_start_var.get(),
            "skip_end": self.skip_end_var.get(),
            "frame_step": self.step_var.get(),
            # Target
            "reference_dir": self.ref_dir_var.get() if self.use_target_var.get() else "",
            "sim_threshold": self.sim_thresh_var.get(),
            "multi_target": self.multi_target_var.get(),
            "keep_non_target": self.keep_non_target_var.get(),
            "reduce_non_target": self.reduce_non_target_var.get(),
            "non_target_keep_interval": self.non_target_keep_interval_var.get(),
            "auto_augment_samples": self.auto_augment_var.get(),
            # Filters
            "blur_classify": self.rt_blur_var.get(),
            "blur_thresholds": [self.rt_blur_low_var.get(), self.rt_blur_high_var.get()],
            "select_mouth_open": self.rt_mouth_var.get(),
            "mouth_open_threshold": self.rt_mouth_thresh_var.get(),
            "classify_pose": self.rt_pose_var.get(),
            "pose_thresholds": [self.rt_pitch_var.get(), self.rt_yaw_var.get()],
            "target_pose_type": self.rt_pose_dir_var.get() if self.rt_pose_var.get() else None,
            "pose_pitch_threshold": self.rt_pitch_var.get(),
            "pose_yaw_threshold": self.rt_yaw_var.get(),
        }
        
        if self.use_target_var.get() and not args["reference_dir"]:
            return None, "启用指定人物提取时必须选择参考目录"
            
        return args, None

    def run_extract(self):
        args, error = self.build_extract_args()
        if error:
            messagebox.showerror("错误", error)
            return
        
        self.log("正在启动提取任务，请稍候...", "info")
        self.run_worker(extractor.run, args)

    def run_batch_extract(self):
        input_dir = self.ex_input_var.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("错误", "批量提取需要选择一个有效的输入目录")
            return
            
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.ts', '.m4v'}
        files = []
        try:
            for f in sorted(os.listdir(input_dir)):
                if os.path.splitext(f)[1].lower() in video_exts:
                    files.append(os.path.join(input_dir, f))
        except Exception as e:
            messagebox.showerror("错误", f"扫描目录失败: {e}")
            return
            
        if not files:
            messagebox.showwarning("提示", "目录中未找到视频文件")
            return
            
        if not messagebox.askyesno("确认", f"找到 {len(files)} 个视频文件，是否开始批量提取？"):
            return
            
        self.log(f"Starting batch extraction for {len(files)} files...", 'info')
        batch_args_list = []
        total = len(files)
        for idx, f in enumerate(files, start=1):
            args, _ = self.build_extract_args(input_path=f)
            if args:
                args["batch_index"] = idx
                args["batch_total"] = total
                batch_args_list.append(args)
        
        self.run_worker(run_batch_worker, batch_args_list)

    def run_filter(self):
        if not self.filter_input_var.get():
            messagebox.showerror("错误", "请输入输入路径")
            return
        
        if self.filter_mode_var.get() == "target" and not self.filter_ref_dir_var.get():
            messagebox.showerror("错误", "指定人物筛选需要选择参考人脸目录")
            return
            
        args = {
            "input_dir": self.filter_input_var.get(),
            "output_dir": self.filter_output_var.get(),
            "mode": self.filter_mode_var.get(),
            "select_poses": self.pose_var.get() if self.filter_mode_var.get() == 'pose' else "",
            "pose_pitch_threshold": self.pitch_var.get(),
            "pose_yaw_threshold": self.yaw_var.get(),
            "mouth_open_threshold": self.mouth_thresh_var.get(),
            "blur_low_threshold": self.blur_low_var.get(),
            "blur_high_threshold": self.blur_high_var.get(),
            "select_mouth_open": (self.filter_mode_var.get() == 'mouth'),
            "blur_analysis": (self.filter_mode_var.get() == 'blur'),
            "reference_dir": self.filter_ref_dir_var.get() if self.filter_mode_var.get() == "target" else "",
            "sim_threshold": self.filter_sim_thresh_var.get(),
            "multi_target": self.filter_multi_target_var.get(),
            "target_pick": "target" if self.filter_target_pick_var.get() else "non_target",
            "debug": False # Could add checkbox
        }
        
        self.run_worker(filter.run, args)

def run_batch_worker(batch_args, log_queue):
    # This runs in a subprocess
    utils.setup_process_logging(log_queue)
    total = len(batch_args)
    for i, args in enumerate(batch_args):
        print(f"Batch Processing [{i+1}/{total}]: {args['input_dir']}")
        try:
            extractor.run(args, log_queue=None) # log_queue already handled globally for this process
        except Exception as e:
            print(f"Error processing {args['input_dir']}: {e}")
        finally:
            try:
                extractor.nn.reset_session()
            except Exception:
                pass
            
def main():
    multiprocessing.freeze_support()
    app = ExtractUI()
    app.mainloop()
