# CCTV AI People Counting V2 (Line Crossing & Clean UI)

Sistem **People Counting** berbasis **YOLO11s + OpenVINO (Intel Iris Xe GPU / CPU)** dan **ByteTrack** untuk menghitung jumlah nasabah yang masuk ke area kursi pelayanan secara *real-time* melalui stream RTSP MediaMTX.

## Fitur Utama V2
- **Line Crossing 2 Titik (setup_line.py):** Bebas menentukan garis batas masuk pelayanan beserta arah panah masuk di setiap kamera (CAM1 - CAM8).
- **Clean UI Mode (main.py):** Saat berjalan di layar utama, garis pelayanan dan kotak deteksi disembunyikan (100% transparan) sehingga layar bersih dan hanya menampilkan **ID** di atas kepala.
- **Hardware Acceleration (intel:gpu / cpu):** Dioptimalkan menggunakan format **OpenVINO FP16** (imgsz=640) untuk deteksi tajam pada orang duduk maupun berdiri.
- **Anti-Double Count & Seat Memory:** Dilengkapi *Spatial Cooldown*, *3-Frame Confirmation Filter* (anti objek non-manusia), dan pengunci memori kursi.
- **3-Thread Asynchronous Architecture:** Stream video RTSP berjalan mulus tanpa *delay* terpisah dari thread pemrosesan AI.

## Struktur Project
\\\	ext
People-Counting/
├── config.py              # Konfigurasi kamera RTSP, model, resolusi, dan threshold
├── custom_tracker.yaml    # Konfigurasi parameter ByteTrack
├── setup_line.py          # Tool interaktif pembuat garis pelayanan per kamera
├── main.py                # Program utama People Counting (Clean UI)
├── requirements.txt       # Daftar library Python
└── data/
    ├── lines/             # Koordinat garis per kamera (cam1.json - cam8.json)
    └── customer_history.json # Penyimpanan otomatis total counter harian (diabaikan di Git)
\\\

## Cara Instalasi (Setup Awal)

### 1. Clone Repository
\\\powershell
git clone https://github.com/Reinaldiabang777/People-Counting.git
cd People-Counting
\\\

### 2. Buat & Aktifkan Virtual Environment
\\\powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
\\\

### 3. Install Library
\\\powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
\\\

### 4. Export Model YOLO11s ke OpenVINO
\\\powershell
yolo export model=yolo11s.pt format=openvino imgsz=640 half=True
\\\

## Cara Penggunaan

### 1. Mengatur Garis Pelayanan (setup_line.py)
Pastikan **MediaMTX** sudah aktif, lalu jalankan:
\\\powershell
python setup_line.py
\\\
- **Klik Kiri 2 Titik:** Membuat garis batas pelayanan.
- **Tombol F:** Membalik arah panah hijau (arah masuk pelayanan).
- **Tombol R:** Reset titik garis.
- **Tombol S:** Simpan garis dan keluar.

### 2. Menjalankan Sistem Utama (main.py)
\\\powershell
python main.py
\\\
**Shortcut Tombol Saat Live Monitoring:**
- **L:** Tampilkan / Sembunyikan garis pelayanan & kotak ROI di layar.
- **F:** Balik arah panah masuk secara langsung & simpan otomatis.
- **Z:** Aktifkan / Nonaktifkan mode *Smart ROI Zoom*.
- **Q:** Keluar dari program.
