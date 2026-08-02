# TASK: Perbaiki CCTV HLS Stream — Speed Tracker AI

> **Untuk AI coding assistant.** Root cause SUDAH terkonfirmasi lewat diagnosa manual.
> Ini bukan dokumen investigasi — ini instruksi implementasi. Kerjakan berurutan.

---

## RINGKASAN MASALAH

Aplikasi deteksi kecepatan kendaraan berbasis CCTV gagal membaca video stream
setelah dideploy ke VPS. Dashboard menampilkan status `RECONNECTING` permanen,
`VideoCapture could not open the stream`, FPS 0, panel video hitam.

**Root cause: server CCTV memakai proteksi cookie dua lapis yang tidak ditangani
oleh FFmpeg bawaan OpenCV.**

---

## BUKTI DIAGNOSA (sudah dilakukan, jangan diulang)

### 1. Log container

```
speed_tracker_app | [WARN] cap_ffmpeg_impl.hpp:453 _opencv_ffmpeg_interrupt_callback
                    Stream timeout triggered after 30324.879142 ms
                    (× ~47 kali dalam rentang 0,1 detik)
speed_tracker_app | [WARN] cap_ffmpeg_impl.hpp:1329 open Unable to read codec
                    parameters from stream (Immediate exit requested)
speed_tracker_app | [WARN] cap.cpp:212 open VIDEOIO(FFMPEG): backend is generally
                    available but can't be used to capture by name
```

Timeout 30 detik (bukan error instan) → koneksi terbentuk tapi data tidak pernah
datang. `Immediate exit requested` adalah efek dari interrupt callback, bukan
codec rusak.

### 2. Request pertama — 302 cookie challenge

```
$ curl -v https://stream-backup.banjarbarukota.go.id/a1_simpang4_arah_pom/index.m3u8

< HTTP/2 302
< server: openresty
< location: /a1_simpang4_arah_pom/index.m3u8?cookieCheck=1
< set-cookie: cookieCheck=1
< set-cookie: cookieCheck=1; HttpOnly; Secure; SameSite=None; Partitioned
```

TLS handshake sukses, sertifikat valid (`*.banjarbarukota.go.id`, Sectigo).
IP VPS **tidak** diblokir.

### 3. Request kedua — dapat master playlist + session cookie

```
$ curl -i -L -b "cookieCheck=1" <url>

< HTTP/2 200
< content-type: application/vnd.apple.mpegurl
< set-cookie: hlsSession=d19569c0-f99d-4128-a472-e48f0caa9283

#EXTM3U
#EXT-X-VERSION:10
#EXT-X-INDEPENDENT-SEGMENTS
#EXT-X-STREAM-INF:BANDWIDTH=811831,CODECS="avc1.4d401f",RESOLUTION=1280x720,FRAME-RATE=20.000
video1_stream.m3u8
```

### 4. Media playlist butuh KEDUA cookie

```
$ curl -s -c /tmp/cj -b /tmp/cj -L <index.m3u8> > /dev/null
$ curl -s -c /tmp/cj -b /tmp/cj -L <video1_stream.m3u8>

#EXTM3U
#EXT-X-TARGETDURATION:3
#EXT-X-MEDIA-SEQUENCE:67642
#EXT-X-MAP:URI="bd245ce82d13_video1_init.mp4"
#EXTINF:2.50769,
bd245ce82d13_video1_seg67642.mp4
```

### Kesimpulan rantai autentikasi

| Langkah | Cookie dibutuhkan | Respons |
|---|---|---|
| `GET index.m3u8` (1) | – | 302 + `cookieCheck=1` |
| `GET index.m3u8` (2) | `cookieCheck` | 200 master playlist + `hlsSession=<uuid>` |
| `GET video1_stream.m3u8` | `cookieCheck` + `hlsSession` | 200 media playlist |
| `GET *_seg*.mp4` | keduanya | segmen fMP4 |

`hlsSession` adalah **UUID acak per sesi** — tidak bisa di-hardcode.
FFmpeg bawaan OpenCV mengabaikan `Set-Cookie` sepenuhnya, sehingga terjebak
redirect loop `302 → 302 → ...` sampai timeout.

### Hipotesis yang SUDAH GUGUR (jangan dikejar lagi)

- ~~IP VPS diblokir / geoblocking~~ → TLS sukses, server merespons normal
- ~~OpenCV tanpa dukungan HTTPS~~ → kalau begitu errornya `Protocol not found` instan
- ~~ffmpeg tidak terpasang~~ → sudah ada di Dockerfile
- ~~Sertifikat / CA bermasalah~~ → `SSL certificate verify ok`

---

## KONTEKS PROJECT

| Item | Nilai |
|---|---|
| Path project | `/home/Medikidz/Job_tracker` |
| Backend | Flask, factory pattern `app:create_app()` |
| WSGI | Gunicorn, 1 worker, 8 threads, timeout 120 |
| AI/CV | Ultralytics YOLO (`weights/yolo26m.pt`), tracker `BOTSORT_CCTV` |
| Database | MySQL 8.0, SQLAlchemy + PyMySQL |
| Proxy | Nginx Alpine, `8080:80` → `app:8000` |
| Base image | `python:3.10-slim` |
| Container | `speed_tracker_app`, `speed_tracker_db`, `speed_tracker_nginx` |
| Jumlah kamera | 69 |
| Resolusi stream | 1280×720 @ 20 fps |

---

## TASK 1 — Buat `hls_stream.py` [WAJIB]

Buat file `hls_stream.py` di root project dengan isi persis seperti berikut.

```python
"""
hls_stream.py — Pembaca stream HLS CCTV Banjarbaru.

Server CCTV (openresty) memakai proteksi cookie dua lapis:
  1. GET index.m3u8            -> 302, Set-Cookie: cookieCheck=1
  2. GET index.m3u8 + cookie   -> 200 master playlist, Set-Cookie: hlsSession=<uuid>
  3. GET video1_stream.m3u8    -> butuh cookieCheck DAN hlsSession
  4. GET *_seg*.mp4            -> butuh keduanya

hlsSession adalah UUID per sesi, tidak bisa di-hardcode. FFmpeg bawaan OpenCV
tidak menangani Set-Cookie, sehingga terjebak redirect loop sampai timeout 30s.

Modul ini melakukan handshake dengan requests.Session untuk menangkap cookie,
lalu meneruskannya ke subprocess ffmpeg lewat opsi -cookies.
"""

import logging
import subprocess
import threading
import time
from urllib.parse import urlparse

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_REFERER = "https://cctv.banjarbarukota.go.id/"


def handshake(url, user_agent=DEFAULT_UA, referer=DEFAULT_REFERER, timeout=15):
    """Handshake cookie. Return (cookies_dict, master_playlist_text)."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Referer": referer,
        "Accept": "*/*",
    })

    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    cookies = session.cookies.get_dict()
    logger.info("Handshake sukses. Cookie: %s", list(cookies.keys()))

    if "hlsSession" not in cookies:
        logger.warning(
            "hlsSession tidak ditemukan. Server mungkin mengubah mekanisme "
            "proteksi. Cookie yang ada: %s", cookies
        )

    return cookies, resp.text


def build_cookie_arg(url, cookies):
    """
    Susun argumen -cookies untuk ffmpeg.
    Format wajib: "name=value; path=/; domain=<host>", dipisah newline.
    Tanpa path dan domain, ffmpeg mengabaikan cookie secara diam-diam.
    """
    host = urlparse(url).hostname or ""
    return "\n".join(
        f"{name}={value}; path=/; domain={host}"
        for name, value in cookies.items()
    )


class CCTVStream:
    """
    Baca frame BGR dari stream HLS berproteksi cookie.
    API kompatibel dengan cv2.VideoCapture: open / read / isOpened / release.
    """

    def __init__(
        self,
        url,
        width=1280,
        height=720,
        timeout=15,
        user_agent=DEFAULT_UA,
        referer=DEFAULT_REFERER,
        auto_reconnect=True,
        max_reconnect=5,
    ):
        self.url = url
        self.width = width
        self.height = height
        self.timeout = timeout
        self.user_agent = user_agent
        self.referer = referer
        self.auto_reconnect = auto_reconnect
        self.max_reconnect = max_reconnect

        self.frame_size = width * height * 3
        self.proc = None
        self.cookies = None
        self._last_error = ""
        self._reconnect_count = 0
        self._lock = threading.Lock()

    def _build_cmd(self):
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            # Kunci perbaikan: cookie hasil handshake
            "-cookies", build_cookie_arg(self.url, self.cookies),
            "-user_agent", self.user_agent,
            "-headers", f"Referer: {self.referer}\r\n",
            # Redirect & timeout
            "-follow_redirects", "1",
            "-rw_timeout", str(self.timeout * 1_000_000),
            # Reconnect kalau stream putus di tengah
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_delay_max", "5",
            # Mulai dari segmen terbaru
            "-live_start_index", "-1",
            "-i", self.url,
            # Output raw BGR ke stdout
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.width}:{self.height}",
            "-an", "-sn",
            "-",
        ]

    def _drain_stderr(self):
        """Kuras stderr di thread terpisah agar pipe tidak penuh dan blocking."""
        proc = self.proc
        if not proc or not proc.stderr:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._last_error = line
                logger.debug("[ffmpeg] %s", line)

    def open(self):
        """Handshake cookie lalu jalankan ffmpeg. Return self."""
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return self

            self.cookies, _ = handshake(
                self.url,
                user_agent=self.user_agent,
                referer=self.referer,
                timeout=self.timeout,
            )

            self.proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size * 4,
            )
            threading.Thread(target=self._drain_stderr, daemon=True).start()
            logger.info("Stream dibuka: %s", self.url)
            return self

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def read(self):
        """Return (ok, frame) seperti cv2.VideoCapture.read()."""
        if not self.isOpened():
            if not (self.auto_reconnect and self._try_reconnect()):
                return False, None

        buf = b""
        while len(buf) < self.frame_size:
            chunk = self.proc.stdout.read(self.frame_size - len(buf))
            if not chunk:
                logger.warning(
                    "Stream terputus. Error terakhir ffmpeg: %s",
                    self._last_error or "(tidak ada)",
                )
                if self.auto_reconnect and self._try_reconnect():
                    return self.read()
                return False, None
            buf += chunk

        self._reconnect_count = 0
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        return True, frame.copy()

    def _try_reconnect(self):
        """Handshake ulang dengan backoff. Return True kalau berhasil."""
        if self._reconnect_count >= self.max_reconnect:
            logger.error(
                "Batas reconnect (%d) tercapai untuk %s",
                self.max_reconnect, self.url,
            )
            return False

        self._reconnect_count += 1
        wait = min(2 ** self._reconnect_count, 30)
        logger.info(
            "Reconnect ke-%d dalam %ds — sesi HLS akan diperbarui",
            self._reconnect_count, wait,
        )
        self._cleanup_proc()
        time.sleep(wait)

        try:
            self.open()
            return self.isOpened()
        except requests.RequestException as exc:
            logger.error("Handshake gagal saat reconnect: %s", exc)
            return False

    def _cleanup_proc(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.proc = None

    def release(self):
        with self._lock:
            self._cleanup_proc()
        logger.info("Stream ditutup: %s", self.url)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()


def open_with_retry(url, retries=3, backoff=3, **kwargs):
    """
    Buka stream dengan retry ber-backoff.
    JANGAN pakai retry loop tanpa jeda — log sebelumnya menunjukkan ~47
    VideoCapture dibuka dalam 0,1 detik.
    """
    for attempt in range(1, retries + 1):
        try:
            stream = CCTVStream(url, **kwargs).open()
            time.sleep(2)
            ok, _ = stream.read()
            if ok:
                logger.info("Stream siap pada percobaan %d", attempt)
                return stream
            stream.release()
        except requests.RequestException as exc:
            logger.warning("Percobaan %d — handshake gagal: %s", attempt, exc)
        except Exception:
            logger.exception("Percobaan %d — error tak terduga", attempt)

        if attempt < retries:
            time.sleep(backoff * attempt)

    logger.error("Gagal membuka stream setelah %d percobaan: %s", retries, url)
    return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    test_url = sys.argv[1] if len(sys.argv) > 1 else (
        "https://stream-backup.banjarbarukota.go.id/"
        "a1_simpang4_arah_pom/index.m3u8"
    )

    print("=" * 60)
    print("TAHAP 1 — Handshake cookie")
    print("=" * 60)
    try:
        cookies, playlist = handshake(test_url)
        print("  Cookie :", cookies)
        print("  Master playlist:")
        for line in playlist.strip().splitlines()[:8]:
            print("   ", line)
    except Exception as exc:
        print("  GAGAL:", exc)
        sys.exit(1)

    print()
    print("=" * 60)
    print("TAHAP 2 — Baca 10 frame")
    print("=" * 60)
    cap = open_with_retry(test_url)
    if cap is None:
        print("  GAGAL membuka stream")
        sys.exit(1)

    t0 = time.time()
    count = 0
    for i in range(10):
        ok, frame = cap.read()
        print(f"  frame {i}: ok={ok} shape={frame.shape if ok else None}")
        if not ok:
            break
        count += 1
    cap.release()

    elapsed = time.time() - t0
    if count:
        print(f"\n  {count} frame dalam {elapsed:.1f}s (~{count / elapsed:.1f} fps)")
    print("  SELESAI")
```

**Verifikasi:** file harus 384 baris. Cek dengan `wc -l hls_stream.py`.

---

## TASK 2 — Pastikan `requests` terpasang [WAJIB]

Cek `requirements.txt`. Kalau `requests` belum ada, tambahkan:

```
requests>=2.31.0
```

(Kemungkinan besar sudah ada karena ultralytics memerlukannya — verifikasi dulu
dengan `pip list | grep -i requests` di dalam container, jangan asal tambah.)

---

## TASK 3 — Tes standalone [GERBANG — JANGAN LEWATI]

```bash
sudo docker cp hls_stream.py speed_tracker_app:/app/
sudo docker exec -it speed_tracker_app python hls_stream.py
```

**Output yang diharapkan:**

```
TAHAP 1 — Handshake cookie
  Cookie : {'cookieCheck': '1', 'hlsSession': 'd19569c0-...'}
  Master playlist:
    #EXTM3U
    #EXT-X-VERSION:10
    ...
TAHAP 2 — Baca 10 frame
  frame 0: ok=True shape=(720, 1280, 3)
  ...
  10 frame dalam 0.6s (~16.7 fps)
  SELESAI
```

**Kalau gagal di sini, JANGAN lanjut ke TASK 4.** Laporkan output lengkapnya
dan tunggu instruksi. Mengubah kode aplikasi sebelum reader terbukti jalan
akan mempersulit pelacakan penyebab.

Kemungkinan kegagalan dan artinya:

| Gejala | Arti | Tindakan |
|---|---|---|
| `ModuleNotFoundError: requests` | Task 2 belum jalan | Install requests |
| `hlsSession tidak ditemukan` | Server ubah mekanisme | Laporkan, perlu diagnosa ulang |
| `frame 0: ok=False` | ffmpeg gagal baca | Ubah `-loglevel warning` → `debug`, laporkan output |
| `ffmpeg: not found` | Binary hilang dari image | Cek Dockerfile, rebuild |

---

## TASK 4 — Ganti `cv2.VideoCapture` [setelah TASK 3 lolos]

Cari semua pemanggilan:

```bash
grep -rn "VideoCapture" --include="*.py" .
```

Ganti setiap pemanggilan yang menargetkan **stream HLS** (URL `.m3u8`):

```python
# SEBELUM
cap = cv2.VideoCapture(url)

# SESUDAH
from hls_stream import CCTVStream
cap = CCTVStream(url, width=1280, height=720).open()
```

**Aturan penting:**

1. **Jangan ubah logika deteksi.** `read()` mengembalikan array BGR
   `(720, 1280, 3)` dengan format identik dengan OpenCV. YOLO, tracker,
   perhitungan kecepatan, dan ROI tidak perlu disentuh.
2. **Jangan hapus `import cv2`.** OpenCV masih dipakai untuk operasi frame
   (`cv2.rectangle`, `cv2.putText`, `cv2.imencode`, dll). Yang diganti
   hanya bagian pembacaan stream.
3. **Jangan ganti `VideoCapture` untuk file video lokal atau webcam** — kalau
   ada, biarkan tetap pakai `cv2.VideoCapture`.
4. Kalau ada pemanggilan `cap.set(cv2.CAP_PROP_*)`, hapus — `CCTVStream`
   tidak punya method `set()`. Resolusi diatur lewat parameter constructor.

---

## TASK 5 — Perbaiki Nginx buffering [WAJIB]

Gejala "tab browser loading terus-menerus" disebabkan `proxy_buffering` aktif
secara default. Untuk MJPEG (`multipart/x-mixed-replace`), buffering menahan
frame di memori dan tidak pernah meneruskannya ke browser.

Backup dulu: `cp nginx/default.conf nginx/default.conf.bak`

Ganti isi `nginx/default.conf`:

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
    # proxy_buffering off WAJIB, tanpa ini video tidak pernah muncul.
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

    # ===== SSE / WebSocket untuk update statistik realtime =====
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

**Sebelum apply**, verifikasi nama route streaming yang sebenarnya:

```bash
grep -rn "multipart/x-mixed-replace" --include="*.py" .
```

Kalau route-nya bernama lain (misal `/live`, `/feed/<camera_id>`), tambahkan
ke regex `location ~ ^/(...)`.

Apply:
```bash
sudo docker compose restart nginx
sudo docker exec speed_tracker_nginx nginx -t
```

---

## TASK 6 — Perbaiki retry storm [WAJIB]

Log menunjukkan ~47 `VideoCapture` dibuka dalam rentang 0,1 detik, semuanya
timeout bersamaan. Ini menandakan salah satu dari:

- Retry loop tanpa jeda (`while True: cap = VideoCapture(url)`)
- Semua 69 kamera dicoba dibuka serentak saat startup

Cari pola tersebut dan perbaiki:

1. **Hanya buka stream untuk kamera yang sedang aktif dilihat user**, bukan
   semua 69 sekaligus.
2. **Gunakan `open_with_retry()`** yang sudah punya backoff, bukan loop manual.
3. **Tandai kamera OFFLINE setelah N kegagalan**, jangan retry selamanya.
4. Kalau memang perlu multi-kamera, pakai worker pool dengan batas
   maksimum 2–4 stream bersamaan.

Alasan teknis: satu container CPU-only tidak akan sanggup menjalankan YOLO
inference untuk 69 stream 1280×720 @ 20fps secara paralel. Ini bukan sekadar
masalah log yang berisik.

---

## TASK 7 — Perbaiki metrik CPU/GPU kosong [OPSIONAL]

Dashboard menampilkan `–` untuk CPU USAGE dan GPU USAGE. Pastikan `psutil`
ada di `requirements.txt`, dan endpoint metrik tidak menelan exception diam-diam:

```python
import psutil

@app.route("/api/system-stats")
def system_stats():
    try:
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_percent": None,   # None = tidak tersedia, bukan error
        }
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu_percent"] = torch.cuda.utilization()
        except Exception:
            pass   # VPS tanpa GPU — ini normal
        return jsonify(stats)
    except Exception as exc:
        app.logger.exception("Gagal mengambil system stats")
        return jsonify({"error": str(exc)}), 500
```

Catatan: `GPU USAGE: –` wajar kalau VPS tidak punya GPU.
`CPU USAGE: –` selalu menandakan bug.

---

## TASK 8 — Perbaikan keamanan [PENTING]

Di `docker-compose.yml`:

### 8.1 Tutup port MySQL

```yaml
ports:
  - "3306:3306"      # HAPUS baris ini
```

Port ini mengekspos MySQL ke seluruh internet. Container `app` mengakses `db`
lewat jaringan internal Docker, jadi mapping ini tidak diperlukan.
Kalau butuh akses eksternal untuk debugging, batasi: `"127.0.0.1:3306:3306"`.

### 8.2 Pindahkan kredensial ke `.env`

`MYSQL_ROOT_PASSWORD: rootpassword` tertulis langsung di compose file.
Pindahkan ke `.env`, tambahkan `.env` ke `.gitignore`, dan gunakan user MySQL
non-root untuk aplikasi.

### 8.3 Hapus atribut `version`

```yaml
version: '3.8'     # HAPUS — deprecated di Compose V2
```

---

## KRITERIA SELESAI

- [ ] `python hls_stream.py` menampilkan `frame 0: ok=True shape=(720, 1280, 3)`
- [ ] `docker compose logs app` bersih dari `Stream timeout triggered`
- [ ] Status kamera di dashboard: `CONNECTED` (bukan `RECONNECTING`)
- [ ] Video terlihat di panel Live CCTV
- [ ] `FPS` > 0
- [ ] `INFERENCE TIME` > 0 ms
- [ ] Bounding box kendaraan muncul di atas frame
- [ ] Spinner loading browser berhenti setelah halaman selesai dimuat
- [ ] `CPU USAGE` menampilkan angka, bukan `–`
- [ ] Tidak ada lonjakan puluhan koneksi serentak di log
- [ ] Data deteksi tersimpan ke MySQL

Verifikasi database:
```bash
sudo docker exec -it speed_tracker_db mysql -uroot -prootpassword speed_detection \
  -e "SHOW TABLES; SELECT COUNT(*) FROM <nama_tabel_deteksi>;"
```

---

## URUTAN EKSEKUSI

```
TASK 1  Buat hls_stream.py
TASK 2  Pastikan requests terpasang
TASK 3  Tes standalone          <- GERBANG, jangan lanjut kalau gagal
TASK 4  Ganti cv2.VideoCapture
TASK 5  Perbaiki nginx
TASK 6  Perbaiki retry storm
TASK 7  Metrik CPU/GPU (opsional)
TASK 8  Keamanan
        Verifikasi kriteria selesai
```

---

## PERINTAH REFERENSI

```bash
# Log realtime
sudo docker compose logs -f app

# Shell ke container
sudo docker exec -it speed_tracker_app bash

# Rebuild bersih
sudo docker compose down
sudo docker compose build --no-cache
sudo docker compose up -d

# Tes stream manual
sudo docker exec -it speed_tracker_app python hls_stream.py

# Validasi nginx
sudo docker exec speed_tracker_nginx nginx -t

# Cek cookie handshake manual
sudo docker exec -it speed_tracker_app curl -i -L -b "cookieCheck=1" \
  "https://stream-backup.banjarbarukota.go.id/a1_simpang4_arah_pom/index.m3u8"
```

---

## CATATAN UNTUK AI IDE

- Root cause **sudah terbukti**. Jangan mengulang diagnosa atau mengusulkan
  hipotesis yang sudah gugur di bagian atas dokumen.
- TASK 3 adalah gerbang. Jangan mengubah kode aplikasi sebelum reader terbukti
  bisa membaca frame.
- Jangan mengganti paket OpenCV — masalahnya bukan di sana.
- `hlsSession` kemungkinan punya masa berlaku. Kalau stream jalan lalu putus
  setelah beberapa menit, itu sesi kedaluwarsa; `auto_reconnect` sudah
  menanganinya, tapi perhatikan frekuensinya di log.
