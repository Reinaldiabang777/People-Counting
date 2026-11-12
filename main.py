import cv2
import json
import time
import threading
import os
import math
import warnings
import logging
from datetime import date
from collections import deque
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from config import (
    MODEL_PATH, TRACKER_CONFIG, IMG_SIZE, CONF_THRESH, IOU_THRESH, DEVICE,
    FRAME_WIDTH, FRAME_HEIGHT, HEAD_BOX_ONLY, PERSIST_DISPLAY_SEC,
    CROSS_COOLDOWN_SECONDS, CROSS_COOLDOWN_RADIUS,
    HISTORY_FILE, CAMERAS
)

warnings.filterwarnings("ignore")
LOGGER.setLevel(logging.ERROR)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


class DailyCounter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.today = str(date.today())
        self.total_today = 0
        self.qualified_ids = set()
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == self.today:
                    self.total_today = data.get("total", 0)
                    return
            except Exception:
                pass
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"date": self.today, "total": self.total_today}, f, indent=2)

    def check_date_reset(self):
        now_day = str(date.today())
        if now_day != self.today:
            self.today = now_day
            self.total_today = 0
            self.qualified_ids.clear()
            self._save()

    def add_customer(self, track_id):
        if track_id not in self.qualified_ids:
            self.qualified_ids.add(track_id)
            self.total_today += 1
            self._save()
            print(f"[COUNTED] ID {track_id:02d} masuk pelayanan! Total hari ini: {self.total_today}")


def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(p1, p2, q1, q2):
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def is_crossing_in(line_p1, line_p2, prev_pt, curr_pt):
    if not segments_intersect(line_p1, line_p2, prev_pt, curr_pt):
        return False
    side_prev = cross_product(line_p1, line_p2, prev_pt)
    side_curr = cross_product(line_p1, line_p2, curr_pt)
    return side_prev < 0 and side_curr >= 0


print("\n==============================")
print("   CCTV PEOPLE COUNTING V2")
print("==============================")
for key, cam in CAMERAS.items():
    print(f"{key}. {cam['name']}")
print("==============================")

choice = input("Pilih kamera [1-8]: ").strip()
if choice not in CAMERAS:
    print("ERROR: Pilihan tidak valid.")
    raise SystemExit(1)

CAM_CFG = CAMERAS[choice]
CAM_NAME = CAM_CFG["name"]
RTSP_URL = CAM_CFG["rtsp"]
LINE_FILE = CAM_CFG["line_file"]

if not os.path.exists(LINE_FILE):
    print(f"ERROR: Garis untuk {CAM_NAME} belum dibuat! Jalankan: python setup_line.py")
    raise SystemExit(1)

with open(LINE_FILE, "r", encoding="utf-8") as f:
    pts_cfg = json.load(f).get("customer_line", [])

if len(pts_cfg) != 2:
    print("ERROR: Garis belum valid. Jalankan ulang: python setup_line.py")
    raise SystemExit(1)

LINE_P1 = tuple(pts_cfg[0])
LINE_P2 = tuple(pts_cfg[1])


def calc_arrow_and_roi(p1, p2):
    mx = int((p1[0] + p2[0]) / 2)
    my = int((p1[1] + p2[1]) / 2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    end = (int(mx + (-dy / length) * 40), int(my + (dx / length) * 40)) if length > 0 else (mx, my)

    min_x = max(0, min(p1[0], p2[0]) - 120)
    max_x = min(FRAME_WIDTH, max(p1[0], p2[0]) + 280)
    min_y = max(0, min(p1[1], p2[1]) - 130)
    max_y = min(FRAME_HEIGHT, max(p1[1], p2[1]) + 120)
    return mx, my, length, end, (min_x, min_y, max_x, max_y)


MID_X, MID_Y, LINE_LEN, ARROW_END, ROI_BOX = calc_arrow_and_roi(LINE_P1, LINE_P2)

print(f"Memuat Model AI ({MODEL_PATH}) dengan imgsz={IMG_SIZE}...")
model = YOLO(MODEL_PATH, task="detect")
counter = DailyCounter(HISTORY_FILE)

latest_raw_frame = None
frame_seq = 0
frame_lock = threading.Lock()

ai_lock = threading.Lock()
track_history = {}
track_first_pt = {}
track_last_seen = {}
track_last_box = {}
track_hits = {}
recent_crossings = deque(maxlen=30)
locked_seat_memory = {}
flash_green_until = 0.0
ai_fps = 25.0

running = True
capture_connected = False
use_smart_roi = True
show_line_overlay = False  # Default: Garis & ROI 100% Transparan (Tersembunyi) di Main


def capture_loop():
    global latest_raw_frame, frame_seq, running, capture_connected
    cap = None

    while running:
        if cap is None:
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cap.release()
                cap = None
                capture_connected = False
                time.sleep(1.5)
                continue
            capture_connected = True
            print(f"RTSP {CAM_NAME} terhubung.")

        ret, raw = cap.read()
        if not ret:
            capture_connected = False
            cap.release()
            cap = None
            time.sleep(0.5)
            continue

        with frame_lock:
            latest_raw_frame = raw
            frame_seq += 1

        time.sleep(0.005)

    if cap is not None:
        cap.release()


def ai_worker_loop():
    global flash_green_until, ai_fps
    last_seq = -1
    last_ai_time = time.time()

    while running:
        with frame_lock:
            if latest_raw_frame is None or frame_seq == last_seq:
                raw_for_ai = None
            else:
                raw_for_ai = latest_raw_frame
                last_seq = frame_seq

        if raw_for_ai is None:
            time.sleep(0.005)
            continue

        full_resized = cv2.resize(raw_for_ai, (FRAME_WIDTH, FRAME_HEIGHT))

        if use_smart_roi:
            rx1, ry1, rx2, ry2 = ROI_BOX
            ai_input = full_resized[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
        else:
            ai_input = full_resized
            offset_x, offset_y = 0, 0

        now = time.time()
        dt = max(now - last_ai_time, 1e-4)
        last_ai_time = now
        ai_fps = 0.85 * ai_fps + 0.15 * (1.0 / dt)

        results = model.track(
            ai_input,
            persist=True,
            classes=[0],
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            tracker=TRACKER_CONFIG,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        boxes = results[0].boxes
        with ai_lock:
            active_centers = []
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()

                for track_id, box in zip(ids, xyxy):
                    x1 = int(box[0]) + offset_x
                    y1 = int(box[1]) + offset_y
                    x2 = int(box[2]) + offset_x
                    y2 = int(box[3]) + offset_y
                    w = x2 - x1
                    h = y2 - y1

                    aspect = h / max(w, 1)
                    area = w * h
                    if w < 28 or h < 45 or aspect < 0.82 or aspect > 4.2 or w > 210 or area < 1400:
                        continue

                    anchor_x = int((x1 + x2) / 2)
                    anchor_y = int(y1 + 0.18 * h)
                    curr_pt = (anchor_x, anchor_y)

                    too_close = False
                    for acx, acy in active_centers:
                        if math.hypot(curr_pt[0] - acx, curr_pt[1] - acy) < 18:
                            too_close = True
                            break
                    if too_close:
                        continue
                    active_centers.append(curr_pt)

                    if HEAD_BOX_ONLY:
                        half_w = min(max(int(w * 0.24), 15), 28)
                        head_h = min(max(int(h * 0.28), 26), 46)
                        bx1 = max(x1, anchor_x - half_w)
                        by1 = y1
                        bx2 = min(x2, anchor_x + half_w)
                        by2 = y1 + head_h
                    else:
                        bx1, by1, bx2, by2 = x1, y1, x2, y2

                    if track_id in track_last_box:
                        ox1, oy1, ox2, oy2, opt = track_last_box[track_id]
                        move_dist = math.hypot(curr_pt[0] - opt[0], curr_pt[1] - opt[1])
                        if move_dist < 5.0:
                            bx1 = int(0.45 * ox1 + 0.55 * bx1)
                            by1 = int(0.45 * oy1 + 0.55 * by1)
                            bx2 = int(0.45 * ox2 + 0.55 * bx2)
                            by2 = int(0.45 * oy2 + 0.55 * by2)
                            curr_pt = (int(0.45 * opt[0] + 0.55 * curr_pt[0]), int(0.45 * opt[1] + 0.55 * curr_pt[1]))

                    track_hits[track_id] = track_hits.get(track_id, 0) + 1
                    track_last_seen[track_id] = now
                    track_last_box[track_id] = (bx1, by1, bx2, by2, curr_pt)

                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=14)
                        track_first_pt[track_id] = curr_pt

                    if len(track_history[track_id]) > 0:
                        last_pt = track_history[track_id][-1]
                        if math.hypot(curr_pt[0] - last_pt[0], curr_pt[1] - last_pt[1]) > 90:
                            track_history[track_id].clear()
                            track_first_pt[track_id] = curr_pt

                    track_history[track_id].append(curr_pt)

                    if track_id not in counter.qualified_ids:
                        for old_id, mem in list(locked_seat_memory.items()):
                            if old_id not in ids and (now - mem["time"]) < 300.0:
                                if math.hypot(curr_pt[0] - mem["pt"][0], curr_pt[1] - mem["pt"][1]) < 45:
                                    counter.qualified_ids.add(track_id)
                                    locked_seat_memory.pop(old_id, None)
                                    break

                    pts = list(track_history[track_id])
                    if len(pts) >= 3 and track_hits.get(track_id, 0) >= 3 and track_id not in counter.qualified_ids:
                        first_pt = track_first_pt.get(track_id, pts[0])
                        crossed = (
                            is_crossing_in(LINE_P1, LINE_P2, pts[0], curr_pt)
                            or is_crossing_in(LINE_P1, LINE_P2, first_pt, curr_pt)
                        )
                        if crossed:
                            is_duplicate = False
                            for cross_time, cross_pt in recent_crossings:
                                if (now - cross_time) < CROSS_COOLDOWN_SECONDS:
                                    if math.hypot(curr_pt[0] - cross_pt[0], curr_pt[1] - cross_pt[1]) < CROSS_COOLDOWN_RADIUS:
                                        is_duplicate = True
                                        break

                            if not is_duplicate:
                                counter.add_customer(track_id)
                                recent_crossings.append((now, curr_pt))
                                flash_green_until = now + 0.6

                    if track_id in counter.qualified_ids:
                        locked_seat_memory[track_id] = {"pt": curr_pt, "time": now}


t_cap = threading.Thread(target=capture_loop, daemon=True)
t_ai = threading.Thread(target=ai_worker_loop, daemon=True)
t_cap.start()

print("Menunggu stream video...")
while latest_raw_frame is None and running:
    time.sleep(0.05)

t_ai.start()

last_display_seq = -1
last_disp_time = time.time()
video_fps = 25.0

print(f"\nSistem aktif di {CAM_NAME}!")
print("- Tekan 'L' untuk Tampilkan/Sembunyikan Garis di layar")
print("- Tekan 'F' untuk balik arah panah hijau")
print("- Tekan 'Z' untuk On/Off mode Smart ROI Zoom")
print("- Tekan 'Q' untuk keluar.\n")

while True:
    counter.check_date_reset()

    with frame_lock:
        if latest_raw_frame is None or frame_seq == last_display_seq:
            raw_disp = None
        else:
            raw_disp = latest_raw_frame
            last_display_seq = frame_seq

    if raw_disp is None:
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        continue

    display_frame = cv2.resize(raw_disp, (FRAME_WIDTH, FRAME_HEIGHT))

    now = time.time()
    dt = max(now - last_disp_time, 1e-4)
    last_disp_time = now
    video_fps = 0.90 * video_fps + 0.10 * (1.0 / dt)

    # Garis & ROI hanya ditampilkan jika tombol 'L' ditekan (Default: 100% Transparan / Tersembunyi)
    if show_line_overlay:
        if use_smart_roi:
            rx1, ry1, rx2, ry2 = ROI_BOX
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (180, 120, 0), 1)
        line_color = (0, 255, 0) if now < flash_green_until else (0, 255, 255)
        cv2.line(display_frame, LINE_P1, LINE_P2, line_color, 2)
        if LINE_LEN > 0:
            cv2.arrowedLine(display_frame, (MID_X, MID_Y), ARROW_END, (0, 255, 0), 2, tipLength=0.35)

    with ai_lock:
        for tid, last_seen_time in list(track_last_seen.items()):
            age = now - last_seen_time
            if age > PERSIST_DISPLAY_SEC:
                if age > 60:
                    track_history.pop(tid, None)
                    track_first_pt.pop(tid, None)
                    track_last_seen.pop(tid, None)
                    track_last_box.pop(tid, None)
                    track_hits.pop(tid, None)
                continue

            if track_hits.get(tid, 0) < 3:
                continue

            bx1, by1, bx2, by2, curr_pt = track_last_box[tid]
            is_counted = tid in counter.qualified_ids
            color = (0, 255, 0) if is_counted else (0, 255, 255)

            # Kotak kepala & titik dihapus (100% Transparan), hanya menampilkan teks ID tepat di atas kepala
            label = f"ID {tid:02d}"
            text_pos = (max(curr_pt[0] - 20, 10), max(by1 - 4, 20))
            # Outline hitam tipis agar teks ID selalu jelas terbaca di background terang maupun gelap
            cv2.putText(display_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4)
            cv2.putText(display_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    cv2.putText(
        display_frame,
        f"QUALIFIED TODAY: {counter.total_today} | VIDEO: {video_fps:.0f} FPS | AI: {ai_fps:.1f} FPS",
        (20, FRAME_HEIGHT - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
    )
    rtsp_status = "RTSP: CONNECTED" if capture_connected else "RTSP: RECONNECTING"
    cv2.putText(
        display_frame, rtsp_status, (20, FRAME_HEIGHT - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if capture_connected else (0, 0, 255), 2
    )

    cv2.imshow("CCTV Counting V2 (Clean UI)", display_frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("l"):
        show_line_overlay = not show_line_overlay
    elif key == ord("f"):
        LINE_P1, LINE_P2 = LINE_P2, LINE_P1
        MID_X, MID_Y, LINE_LEN, ARROW_END, ROI_BOX = calc_arrow_and_roi(LINE_P1, LINE_P2)
        with open(LINE_FILE, "w", encoding="utf-8") as f:
            json.dump({"customer_line": [list(LINE_P1), list(LINE_P2)]}, f, indent=2)
        print("Arah panah berhasil dibalik & disimpan!")
    elif key == ord("z"):
        use_smart_roi = not use_smart_roi
        print(f"Smart ROI Zoom: {'AKTIF' if use_smart_roi else 'NONAKTIF'}")

running = False
t_cap.join(timeout=1)
t_ai.join(timeout=1)
cv2.destroyAllWindows()
