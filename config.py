import os

if os.path.exists("yolo11s_openvino_model"):
    MODEL_PATH = "yolo11s_openvino_model"
elif os.path.exists("yolo11n_openvino_model"):
    MODEL_PATH = "yolo11n_openvino_model"
else:
    MODEL_PATH = "yolo11s.pt"

TRACKER_CONFIG = "custom_tracker.yaml"

# Resolusi 640 + GPU Intel Iris Xe (Anti Objek Non-Manusia)
IMG_SIZE = 640
CONF_THRESH = 0.40
IOU_THRESH = 0.50
DEVICE = "intel:gpu"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

HEAD_BOX_ONLY = True
PERSIST_DISPLAY_SEC = 1.2
CROSS_COOLDOWN_SECONDS = 4.0
CROSS_COOLDOWN_RADIUS = 65

DATA_DIR = "data"
LINES_DIR = os.path.join(DATA_DIR, "lines")
HISTORY_FILE = os.path.join(DATA_DIR, "customer_history.json")

os.makedirs(LINES_DIR, exist_ok=True)

CAMERAS = {
    "1": {"name": "CAM1", "rtsp": "rtsp://127.0.0.1:8954/cam1", "line_file": os.path.join(LINES_DIR, "cam1.json")},
    "2": {"name": "CAM2", "rtsp": "rtsp://127.0.0.1:8954/cam2", "line_file": os.path.join(LINES_DIR, "cam2.json")},
    "3": {"name": "CAM3", "rtsp": "rtsp://127.0.0.1:8954/cam3", "line_file": os.path.join(LINES_DIR, "cam3.json")},
    "4": {"name": "CAM4", "rtsp": "rtsp://127.0.0.1:8954/cam4", "line_file": os.path.join(LINES_DIR, "cam4.json")},
    "5": {"name": "CAM5", "rtsp": "rtsp://127.0.0.1:8954/cam5", "line_file": os.path.join(LINES_DIR, "cam5.json")},
    "6": {"name": "CAM6", "rtsp": "rtsp://127.0.0.1:8954/cam6", "line_file": os.path.join(LINES_DIR, "cam6.json")},
    "7": {"name": "CAM7", "rtsp": "rtsp://127.0.0.1:8954/cam7", "line_file": os.path.join(LINES_DIR, "cam7.json")},
    "8": {"name": "CAM8", "rtsp": "rtsp://127.0.0.1:8954/cam8", "line_file": os.path.join(LINES_DIR, "cam8.json")},
}
