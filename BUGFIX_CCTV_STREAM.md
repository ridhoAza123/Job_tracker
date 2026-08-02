# Bug Report & Fix Guide — Speed Tracker AI (CCTV Stream Tidak Terbaca)

> Dokumen ini ditujukan untuk AI coding assistant di IDE (Cursor / Claude Code / Copilot).
> Berisi konteks lengkap, hipotesis penyebab, langkah diagnosa, dan perbaikan yang harus diterapkan.
> Kerjakan **berurutan**: FASE 1 (diagnosa) → FASE 2 (fix pasti) → FASE 3 (fix kondisional).

---

## 1. Konteks Project

| Item | Nilai |
|---|---|
| Nama aplikasi | Speed Tracker AI — ITMS Monitoring |
| Fungsi | Deteksi kendaraan & pengukuran kecepatan dari CCTV publik |
| Backend | Flask (factory pattern: `app:create_app()`) |
| WSGI | Gunicorn — 1 worker, 8 threads, timeout 120 |
| AI/CV | Ultralytics YOLO (`weights/yolo26m.pt`), tracker `BOTSORT_CCTV` |
| Database | MySQL 8.0 (container `speed_tracker_db`), via SQLAlchemy + PyMySQL |
| Reverse proxy | Nginx Alpine, host port `8080` → container port `80` → `app:8000` |
| Deployment | Docker Compose di VPS |
| Base image | `python:3.10-slim` |

### Container yang berjalan
- `speed_tracker_db` — MySQL
- `speed_tracker_app` — Flask + YOLO
- `speed_tracker_nginx` — reverse proxy

### Environment variable saat ini (dari `docker-compose.yml`)
```
HOST=0.0.0.0
PORT=8000
FLASK_DEBUG=0
DATABASE_URL=mysql+pymysql://root:rootpassword@db:3306/speed_detection
YOLO_WEIGHTS_PATH=weights/yolo26m.pt
CCTV_STREAM_URL=https://cctv.banjarbarukota.go.id/CCTV
```

### Stream yang bermasalah
```
https://stream-backup.banjarbarukota.go.id/a1_simpang4_arah_pom/index.m3u8
```
Format: **HLS (.m3u8) over HTTPS**. Resolusi terdaftar 1280x720.
Total 69 kamera terdaftar di sistem.

---

## 2. Gejala yang Diamati

### Gejala A — Stream tidak terbaca
Dashboard menampilkan:
- Status kamera: `RECONNECTING` (tidak pernah jadi `CONNECTED`)
- Pesan error: **`VideoCapture could not open the stream`**
- `FPS: 0`, `Inference time: 0 ms`
- Panel video hitam total

### Gejala B — Browser loading terus-menerus
Spinner di tab browser berputar tanpa henti. Halaman tetap interaktif,
tapi request streaming tidak pernah selesai.

### Gejala C — Metrik sistem kosong
`CPU USAGE` dan `GPU USAGE` menampilkan `–` (bukan angka, bukan `0`).
Mengindikasikan endpoint metrik gagal atau library monitoring tidak terpasang.

### Catatan penting
Aplikasi ini **berfungsi normal saat dijalankan di lokal**. Masalah muncul
setelah deploy ke VPS. Ini mempersempit penyebab ke hal-hal yang berubah antara
lingkungan lokal dan VPS: alamat IP, isolasi jaringan container, dan lapisan Nginx.

---

## 3. Analisis Akar Masalah

Gejala A dan Gejala B **kemungkinan besar penyebabnya berbeda**. Jangan asumsikan
satu perbaikan menyelesaikan keduanya.

### 3.1 Hipotesis untuk Gejala A (stream tidak terbaca)

**H1 — IP VPS diblokir oleh server CCTV** *(probabilitas: tinggi)*
Server CCTV milik pemerintah daerah umumnya membatasi akses berdasarkan
region IP, atau mewajibkan header `Referer` yang valid. Di laptop lokal
(IP Indonesia, region sama) berhasil; dari VPS (IP datacenter, mungkin luar negeri)
ditolak dengan 403.

**H2 — OpenCV tidak mendukung HTTPS** *(probabilitas: tinggi)*
Ini jebakan yang sering terlewat. Paket `ffmpeg` yang diinstal via `apt-get`
di Dockerfile **tidak digunakan oleh `cv2`**. Wheel `opencv-python` membundel
FFmpeg-nya sendiri, dan build bundle tersebut kadang dikompilasi tanpa dukungan
TLS/HTTPS. Akibatnya `ffprobe` di terminal bisa membaca stream, tetapi
`cv2.VideoCapture()` tetap gagal membuka URL yang sama.

**H3 — Tidak ada capture options (header & timeout)** *(probabilitas: sedang)*
FFmpeg di dalam OpenCV secara default mengirim User-Agent generik tanpa Referer,
dan tanpa timeout eksplisit. Banyak server HLS menolak request semacam ini,
atau koneksi menggantung tanpa batas.

**H4 — Mismatch URL** *(probabilitas: sedang)*
`CCTV_STREAM_URL` di compose menunjuk `cctv.banjarbarukota.go.id`, sedangkan
stream aktual berada di subdomain berbeda: `stream-backup.banjarbarukota.go.id`.
Perlu diverifikasi apakah env var ini masih dipakai atau URL sudah diambil dari database.

### 3.2 Hipotesis untuk Gejala B (loading terus)

**H5 — Nginx `proxy_buffering` aktif** *(probabilitas: sangat tinggi)*
Nginx secara default melakukan buffering pada response dari upstream.
Untuk MJPEG streaming (`multipart/x-mixed-replace`) dan Server-Sent Events,
buffering ini menahan frame di memori dan tidak pernah meneruskannya ke browser.
Gejalanya persis: koneksi terbuka selamanya, tab loading terus, video tidak muncul.
**Ini harus diperbaiki terlepas dari hasil diagnosa Gejala A.**

### 3.3 Hipotesis untuk Gejala C

**H6 — `psutil` tidak terpasang** *(probabilitas: sedang)*
Endpoint metrik kemungkinan melempar exception yang ditangkap secara diam-diam,
sehingga frontend menerima `null` dan menampilkan `–`.

---

## 4. FASE 1 — Diagnosa (WAJIB DIJALANKAN DULU)

Buat file `diagnose_stream.py` di root project:

```python
"""Script diagnosa stream CCTV — dijalankan DI DALAM container app."""
import os
import subprocess
import sys

import cv2

URL = os.getenv(
    "TEST_URL",
    "https://stream-backup.banjarbarukota.go.id/a1_simpang4_arah_pom/index.m3u8",
)

print("=" * 70)
print("1. CEK BUILD OPENCV — apakah FFmpeg aktif?")
print("=" * 70)
for line in cv2.getBuildInformation().splitlines():
    if any(k in line for k in ("FFMPEG", "avcodec", "avformat", "GStreamer")):
        print("   ", line.strip())

print("\n" + "=" * 70)
print("2. CEK KONEKTIVITAS HTTP ke server CCTV")
print("=" * 70)
try:
    out = subprocess.run(
        ["curl", "-sS", "-o", "/dev/null", "-w",
         "http_code=%{http_code} time=%{time_total}s size=%{size_download}\n",
         "--max-time", "15", URL],
        capture_output=True, text=True, timeout=30,
    )
    print("   ", out.stdout.strip() or out.stderr.strip())
except Exception as e:
    print("    GAGAL:", e)

print("\n   Isi playlist (10 baris pertama):")
try:
    out = subprocess.run(["curl", "-sS", "--max-time", "15", URL],
                         capture_output=True, text=True, timeout=30)
    body = out.stdout.strip()
    if body:
        for line in body.splitlines()[:10]:
            print("     ", line)
    else:
        print("      (kosong) stderr:", out.stderr.strip())
except Exception as e:
    print("    GAGAL:", e)

print("\n" + "=" * 70)
print("3. CEK FFPROBE (ffmpeg sistem)")
print("=" * 70)
try:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate",
         "-of", "default=noprint_wrappers=1", URL],
        capture_output=True, text=True, timeout=45,
    )
    print("    stdout:", out.stdout.strip() or "(kosong)")
    print("    stderr:", out.stderr.strip() or "(kosong)")
except Exception as e:
    print("    GAGAL:", e)

print("\n" + "=" * 70)
print("4. CEK cv2.VideoCapture — dengan dan tanpa capture options")
print("=" * 70)

def coba(label, opts=None):
    if opts is not None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    opened = cap.isOpened()
    ok, frame = (False, None)
    if opened:
        ok, frame = cap.read()
    cap.release()
    shape = frame.shape if (ok and frame is not None) else None
    print(f"    [{label}] isOpened={opened} read_ok={ok} shape={shape}")

coba("default")
coba("with_headers",
     "user_agent;Mozilla/5.0 (Windows NT 10.0; Win64; x64)|"
     "referer;https://cctv.banjarbarukota.go.id/|"
     "rw_timeout;15000000|timeout;15000000")

sys.exit(0)
```

Jalankan:
```bash
docker cp diagnose_stream.py speed_tracker_app:/app/
docker exec -it speed_tracker_app python diagnose_stream.py
docker compose logs --tail=100 app
```

### Tabel keputusan

| Hasil diagnosa | Akar masalah | Terapkan |
|---|---|---|
| `http_code=403` / `000` / timeout | H1 — IP diblokir | FIX 4 |
| `[default]` gagal, `[with_headers]` sukses | H3 — kurang header | FIX 2 |
| `ffprobe` sukses, kedua `cv2` gagal | H2 — OpenCV tanpa HTTPS | FIX 3 |
| Build info tidak menyebut `FFMPEG: YES` | H2 — OpenCV tanpa FFmpeg | FIX 3 |
| Semua sukses tapi web tetap hitam | H5 — Nginx buffering | FIX 1 saja |

---

## 5. FASE 2 — FIX 1: Nginx (TERAPKAN TANPA SYARAT)

Perbaikan ini diperlukan apapun hasil diagnosa. Ganti isi `nginx/default.conf`:

```nginx
server {
    listen 80;
    server_name _;
    client_max_body_size 32m;

    location /static/ {
        alias /app/static/;
        expires 7d;
        access_log off;
    }

    # ===== ENDPOINT STREAMING MJPEG =====
    # proxy_buffering off WAJIB. Tanpa ini nginx menahan frame di buffer
    # dan video tidak pernah muncul di browser.
    # SESUAIKAN nama route dengan yang ada di kode Flask.
    location ~ ^/(video_feed|stream|mjpeg) {
        proxy_pass http://app:8000;
        proxy_http_version 1.1;

        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        chunked_transfer_encoding off;

        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
    }

    # ===== SSE / WEBSOCKET untuk update statistik realtime =====
    location ~ ^/(events|sse|socket\.io) {
        proxy_pass http://app:8000;
        proxy_http_version 1.1;

        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location / {
        proxy_pass http://app:8000;
        proxy_http_version 1.1;
        proxy_read_timeout 120s;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**TUGAS untuk AI IDE:**
1. Cari route streaming di kode Flask:
   ```
   grep -rn "multipart/x-mixed-replace\|video_feed\|Response(" --include="*.py" .
   ```
2. Pastikan nama route tersebut tercantum di regex `location ~ ^/(...)`.
   Jika route bernama lain (misal `/live`, `/feed/<id>`), tambahkan ke daftar.
3. Terapkan: `docker compose restart nginx`
4. Verifikasi: `docker exec speed_tracker_nginx nginx -t`

---

## 6. FASE 3 — Fix Kondisional

### FIX 2 — Tambah Capture Options (jika H3 terkonfirmasi)

`OPENCV_FFMPEG_CAPTURE_OPTIONS` **harus di-set sebelum `import cv2`**.
Jika di-set setelahnya, tidak akan berpengaruh.

Buat file `stream_config.py` di root project:

```python
"""
Konfigurasi FFmpeg untuk OpenCV.
PENTING: modul ini WAJIB di-import sebelum `import cv2` di manapun.
"""
import os

_FFMPEG_OPTS = "|".join([
    "user_agent;Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "referer;https://cctv.banjarbarukota.go.id/",
    "rw_timeout;15000000",      # 15 detik, satuan mikrodetik
    "timeout;15000000",
    "reconnect;1",
    "reconnect_streamed;1",
    "reconnect_delay_max;5",
])

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_OPTS)
```

Lalu di **baris paling atas** entrypoint (`app.py`) dan modul detector:

```python
import stream_config  # noqa: F401  <- HARUS sebelum import cv2
import cv2
```

Tambahkan juga guard saat membuka stream:

```python
import time
import logging

logger = logging.getLogger(__name__)

def open_stream(url, retries=3, delay=3):
    """Buka stream dengan retry dan logging yang informatif."""
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # kurangi latency
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                logger.info("Stream terbuka pada percobaan %d: %s", attempt, url)
                return cap
            logger.warning("Percobaan %d: isOpened=True tapi read() gagal", attempt)
        else:
            logger.warning("Percobaan %d: VideoCapture gagal membuka %s", attempt, url)
        cap.release()
        if attempt < retries:
            time.sleep(delay)
    logger.error("Gagal membuka stream setelah %d percobaan: %s", retries, url)
    return None
```

### FIX 3 — Ganti Backend Pembacaan Video (jika H2 terkonfirmasi)

Jika OpenCV memang tidak mendukung HTTPS, ada dua opsi.

**Opsi A — Ganti paket OpenCV** (coba dulu, lebih sederhana):

Di `requirements.txt`, ganti:
```
opencv-python==<versi>
```
menjadi:
```
opencv-contrib-python-headless==4.10.0.84
```

Gunakan varian `headless` karena container tidak punya display server —
ini juga memperkecil ukuran image. Lalu rebuild:
```bash
docker compose build --no-cache app
docker compose up -d
```

**Opsi B — Baca via subprocess FFmpeg** (fallback yang paling andal):

Melewati OpenCV sepenuhnya untuk pembacaan stream. Menggunakan binary `ffmpeg`
sistem yang sudah terpasang di Dockerfile, yang dukungan protokolnya lengkap.

Buat `ffmpeg_reader.py`:

```python
"""Pembaca frame via subprocess FFmpeg — bypass keterbatasan OpenCV."""
import logging
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


class FFmpegStreamReader:
    """Baca frame BGR dari stream HLS/RTSP menggunakan binary ffmpeg."""

    def __init__(self, url, width=1280, height=720, timeout=15):
        self.url = url
        self.width = width
        self.height = height
        self.timeout = timeout
        self.frame_size = width * height * 3
        self.proc = None

    def open(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "-headers", "Referer: https://cctv.banjarbarukota.go.id/\r\n",
            "-rw_timeout", str(self.timeout * 1_000_000),
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", self.url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.width}:{self.height}",
            "-an",              # buang audio
            "-sn",              # buang subtitle
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )
        logger.info("FFmpeg reader dimulai untuk %s", self.url)
        return self

    def read(self):
        """Return (ok, frame). Kompatibel dengan API cv2.VideoCapture.read()."""
        if self.proc is None or self.proc.stdout is None:
            return False, None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        return True, frame

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def release(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()
```

Kelas ini punya signature yang sama dengan `cv2.VideoCapture`
(`read()`, `isOpened()`, `release()`), jadi bisa dipakai sebagai pengganti langsung
tanpa mengubah logika detection loop.

### FIX 4 — Masalah Blokir IP (jika H1 terkonfirmasi)

Ini **bukan bug kode** — tidak bisa diperbaiki dari sisi aplikasi.
Pilihan yang tersedia:

1. **Pindahkan VPS ke provider dengan IP Indonesia**
   (Biznet, IDCloudHost, Niagahoster, DigitalOcean Singapore kadang lolos).
2. **Pasang HTTP proxy di jaringan lokal Banjarbaru**, lalu arahkan
   FFmpeg ke proxy tersebut via `http_proxy` env var.
3. **Jalankan detector di mesin lokal**, kirim hasil deteksi ke VPS
   lewat API. Arsitektur ini justru lebih baik: mengurangi beban bandwidth VPS,
   dan proses YOLO berjalan di dekat sumber stream.
4. **Hubungi Diskominfo Banjarbaru** untuk whitelist IP VPS.

Verifikasi hipotesis ini dengan membandingkan:
```bash
# dari VPS
curl -I https://stream-backup.banjarbarukota.go.id/a1_simpang4_arah_pom/index.m3u8

# dari laptop lokal
curl -I https://stream-backup.banjarbarukota.go.id/a1_simpang4_arah_pom/index.m3u8
```
Jika VPS mengembalikan 403 sementara lokal mengembalikan 200, hipotesis terkonfirmasi.

### FIX 5 — Metrik CPU/GPU Kosong (Gejala C)

Tambahkan ke `requirements.txt`:
```
psutil==6.0.0
```

Pastikan endpoint metrik tidak menelan exception:

```python
import psutil

@app.route("/api/system-stats")
def system_stats():
    try:
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_percent": None,  # None = tidak tersedia, bukan error
        }
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu_percent"] = torch.cuda.utilization()
        except Exception:
            pass  # GPU memang tidak ada di VPS — ini normal
        return jsonify(stats)
    except Exception as exc:
        app.logger.exception("Gagal mengambil system stats")
        return jsonify({"error": str(exc)}), 500
```

Di frontend, bedakan tampilan `null` (tidak tersedia) dari error.
`GPU USAGE: –` itu wajar jika VPS tanpa GPU, tapi `CPU USAGE: –` selalu menandakan bug.

---

## 7. Perbaikan Arsitektur yang Direkomendasikan

Di luar perbaikan bug, ada beberapa masalah struktural yang perlu ditangani.

### 7.1 Detection loop menempel di web worker

Saat ini Gunicorn dijalankan dengan `--workers 1 --threads 8`. Jika loop deteksi YOLO
berjalan di dalam worker tersebut, inference yang berat akan memblokir request HTTP.
Dengan 69 kamera, ini tidak akan bisa diskalakan.

**Rekomendasi:** pisahkan menjadi service `detector` tersendiri di `docker-compose.yml`,
berbagi database dan volume dengan `app`. Web hanya membaca hasil dari DB.

```yaml
  detector:
    build: .
    container_name: speed_tracker_detector
    restart: always
    environment:
      - DATABASE_URL=mysql+pymysql://root:rootpassword@db:3306/speed_detection
      - YOLO_WEIGHTS_PATH=weights/yolo26m.pt
    volumes:
      - ./static:/app/static
      - ./logs:/app/logs
    depends_on:
      db:
        condition: service_healthy
    command: ["python", "detector.py"]
```

### 7.2 Kredensial hardcoded

`rootpassword` tertulis langsung di `docker-compose.yml`. Pindahkan ke file `.env`
dan tambahkan `.env` ke `.gitignore`. Gunakan user MySQL non-root untuk aplikasi.

### 7.3 Port MySQL terbuka ke publik

```yaml
ports:
  - "3306:3306"
```
Ini mengekspos MySQL ke internet. Karena `app` mengakses `db` lewat jaringan internal
Docker, mapping port ini tidak diperlukan. **Hapus baris tersebut**, atau batasi ke
localhost saja: `"127.0.0.1:3306:3306"`.

### 7.4 Versi compose usang

`version: '3.8'` sudah deprecated di Docker Compose V2. Baris ini bisa dihapus.

---

## 8. Kriteria Selesai

Perbaikan dianggap berhasil bila semua terpenuhi:

- [ ] `diagnose_stream.py` menunjukkan `read_ok=True` dengan shape `(720, 1280, 3)`
- [ ] `docker compose logs app` tidak lagi memunculkan `VideoCapture could not open`
- [ ] Status kamera di dashboard berubah dari `RECONNECTING` menjadi `CONNECTED`
- [ ] Video terlihat di panel Live CCTV
- [ ] `FPS` menunjukkan angka > 0
- [ ] `INFERENCE TIME` menunjukkan angka > 0 ms
- [ ] Spinner loading di tab browser berhenti setelah halaman selesai dimuat
- [ ] `CPU USAGE` menampilkan persentase, bukan `–`
- [ ] Bounding box deteksi kendaraan muncul di atas frame video
- [ ] Data terekam ke tabel MySQL (verifikasi: `docker exec -it speed_tracker_db mysql -uroot -prootpassword speed_detection -e "SELECT COUNT(*) FROM <nama_tabel>;"`)

---

## 9. Urutan Eksekusi (Ringkas)

```
1. Jalankan diagnose_stream.py           → tentukan akar masalah
2. Terapkan FIX 1 (nginx)                → tanpa syarat
3. Terapkan FIX 5 (psutil)               → tanpa syarat, murah
4. Terapkan FIX 2/3/4 sesuai tabel §4    → kondisional
5. docker compose up -d --build
6. Cek kriteria §8
7. Jika masih gagal, kirim ulang output diagnose_stream.py + docker compose logs app
```

---

## 10. Perintah Referensi

```bash
# Log realtime
docker compose logs -f app

# Masuk ke shell container
docker exec -it speed_tracker_app bash

# Rebuild bersih
docker compose down && docker compose build --no-cache && docker compose up -d

# Cek konektivitas dari dalam container
docker exec -it speed_tracker_app curl -I <url_stream>

# Validasi konfigurasi nginx
docker exec speed_tracker_nginx nginx -t

# Cek koneksi database
docker exec -it speed_tracker_app python -c "
from sqlalchemy import create_engine, text
import os
e = create_engine(os.environ['DATABASE_URL'])
print(e.connect().execute(text('SELECT VERSION()')).scalar())
"

# Cek dukungan protokol FFmpeg
docker exec -it speed_tracker_app ffmpeg -protocols | grep -i https
```
