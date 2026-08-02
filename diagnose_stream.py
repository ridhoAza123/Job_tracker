"""
Script diagnosa stream CCTV — jalankan DI DALAM container app.
Cara pakai:
    docker compose cp diagnose_stream.py app:/app/
    docker compose exec app python diagnose_stream.py
"""
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
info = cv2.getBuildInformation()
for line in info.splitlines():
    if any(k in line for k in ("FFMPEG", "avcodec", "avformat", "GStreamer")):
        print("   ", line.strip())

print()
print("=" * 70)
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

print()
print("   Isi playlist (10 baris pertama):")
try:
    out = subprocess.run(
        ["curl", "-sS", "--max-time", "15", URL],
        capture_output=True, text=True, timeout=30,
    )
    body = out.stdout.strip()
    if body:
        for line in body.splitlines()[:10]:
            print("     ", line)
    else:
        print("      (kosong) stderr:", out.stderr.strip())
except Exception as e:
    print("    GAGAL:", e)

print()
print("=" * 70)
print("3. CEK FFPROBE (ffmpeg sistem) — bisa baca stream?")
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

print()
print("=" * 70)
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
    return ok

coba("default")
coba(
    "with_headers",
    "user_agent;Mozilla/5.0 (Windows NT 10.0; Win64; x64)|"
    "referer;https://cctv.banjarbarukota.go.id/|"
    "rw_timeout;15000000|timeout;15000000",
)

print()
print("Selesai. Kirimkan seluruh output ini untuk analisa lanjutan.")
sys.exit(0)
