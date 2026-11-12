import cv2
import json
import os
import math
from config import CAMERAS, FRAME_WIDTH, FRAME_HEIGHT

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

print("\n==============================")
print("   SETUP GARIS PELAYANAN V2")
print("==============================")
for key, cam in CAMERAS.items():
    print(f"{key}. {cam['name']}")
print("==============================")

choice = input("Pilih kamera [1-8]: ").strip()
if choice not in CAMERAS:
    print("Pilihan tidak valid!")
    raise SystemExit(1)

cam_cfg = CAMERAS[choice]
rtsp_url = cam_cfg["rtsp"]
line_file = cam_cfg["line_file"]

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print(f"ERROR: Gagal mengambil gambar dari {rtsp_url}. Pastikan MediaMTX sudah jalan!")
    raise SystemExit(1)

frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
points = []

if os.path.exists(line_file):
    try:
        with open(line_file, "r", encoding="utf-8") as f:
            points = [tuple(pt) for pt in json.load(f).get("customer_line", [])]
    except Exception:
        points = []

def mouse_cb(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 2:
            points = []
        points.append((x, y))

cv2.namedWindow("Setup Line")
cv2.setMouseCallback("Setup Line", mouse_cb)

print("\nINSTRUKSI:")
print("- Klik Kiri 2 titik untuk membuat garis batas masuk pelayanan")
print("- Tekan 'F' untuk membalik arah panah hijau (arah MASUK)")
print("- Tekan 'R' untuk reset titik")
print("- Tekan 'S' untuk SIMPAN & keluar")
print("- Tekan 'Q' untuk batal\n")

while True:
    disp = frame.copy()
    for pt in points:
        cv2.circle(disp, pt, 6, (0, 0, 255), -1)

    if len(points) == 2:
        p1, p2 = points[0], points[1]
        cv2.line(disp, p1, p2, (0, 255, 255), 3)
        mx, my = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length > 0:
            end = (int(mx + (-dy / length) * 40), int(my + (dx / length) * 40))
            cv2.arrowedLine(disp, (mx, my), end, (0, 255, 0), 3, tipLength=0.35)

    cv2.putText(disp, "[Klik 2 Titik] | [F: Balik Panah] | [R: Reset] | [S: Simpan] | [Q: Keluar]",
                (15, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
    cv2.imshow("Setup Line", disp)

    key = cv2.waitKey(20) & 0xFF
    if key == ord("r"):
        points = []
    elif key == ord("f") and len(points) == 2:
        points = [points[1], points[0]]
    elif key == ord("s"):
        if len(points) == 2:
            os.makedirs(os.path.dirname(line_file), exist_ok=True)
            with open(line_file, "w", encoding="utf-8") as f:
                json.dump({"customer_line": [list(points[0]), list(points[1])]}, f, indent=2)
            print(f"BERHASIL disimpan ke {line_file}!")
            break
        else:
            print("Harus klik 2 titik dulu sebelum menyimpan!")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
