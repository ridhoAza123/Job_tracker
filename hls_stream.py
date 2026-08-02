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
