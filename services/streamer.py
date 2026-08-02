"""Background video capture.

One ``cv2.VideoCapture`` is created per camera and lives in a dedicated
reader thread, feeding three independent consumers on every successful read:

* a single "latest frame" slot, always overwritten — kept for callers that
  just want whatever is freshest right now (e.g. manual capture);
* a bounded playout ring buffer, drained at a steady pace by the render
  thread, which is what makes the *preview* look smooth (see the module docs
  in ``services/pipeline.py`` and ``Config.PLAYOUT_*`` for why this exists);
* a bounded AI backlog queue, drained strictly FIFO by the AI/tracker
  thread. Accuracy-first: every captured frame is queued and processed in
  order, with no intentional skip to keep up with real time (see
  ``Config.AI_QUEUE_SIZE`` and the module docstring in
  ``services/pipeline.py``).

Reconnection uses a fixed backoff schedule and logs one line per state
transition instead of spamming "opening stream" on every retry.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

import cv2

from config import Config

logger = logging.getLogger(__name__)

# Audited against the live HLS source (see Config.PLAYOUT_BUFFER_FRAMES for
# the full rationale): a bigger internal buffer measurably smooths out
# segment-boundary stalls instead of exposing them as read() latency, so
# CAP_PROP_BUFFERSIZE is intentionally *not* set to 1 here. ``timeout`` /
# ``rw_timeout`` are in microseconds and bound blocking socket reads, which
# is what stops a dead stream from wedging the reader thread indefinitely.
# ``reconnect_at_eof`` specifically helps live HLS playlists that briefly
# have no new segment ready.
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'user_agent;Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    '|referer;https://cctv.banjarbarukota.go.id/'
    '|analyzeduration;1000000'
    '|probesize;1000000'
    '|reconnect;1'
    '|reconnect_streamed;1'
    '|reconnect_at_eof;1'
    '|reconnect_delay_max;5'
    '|timeout;15000000'
    '|rw_timeout;15000000',
)
CAPTURE_BUFFERSIZE = 8

STATE_IDLE = 'IDLE'
STATE_CONNECTING = 'CONNECTING'
STATE_ONLINE = 'ONLINE'
STATE_RECONNECTING = 'RECONNECTING'
STATE_FAILED = 'FAILED'
STATE_STOPPED = 'STOPPED'


class Streamer:
    """Threaded, self-healing single-capture frame source."""

    def __init__(
        self,
        logger_: Optional[logging.Logger] = None,
        on_state_change: Optional[Callable[[str, str], None]] = None,
    ):
        self.logger = logger_ or logger
        self._on_state_change = on_state_change

        self._url: Optional[str] = None

        self._frame = None
        self._frame_id = 0
        self._frame_time = 0.0
        self._frame_lock = threading.Lock()

        # Playout ring buffer: bounded, FIFO, drained by the render thread at
        # a steady pace. Deliberately separate from the "latest frame" slot
        # above — see the module docstring.
        self._playout: deque = deque(maxlen=Config.PLAYOUT_BUFFER_FRAMES)
        self._playout_lock = threading.Lock()

        # AI-side backlog queue: bounded, FIFO, drained sequentially by the
        # AI/tracker loop — every captured frame goes in here and is meant
        # to come back out and be processed, in order, even if that lags
        # behind the live camera (accuracy-first: see Config.AI_QUEUE_SIZE
        # and the module docstring in services/pipeline.py). The bound only
        # exists as a safety valve against unbounded memory growth if
        # processing falls behind capture for a sustained period; hitting it
        # is logged loudly via ai_queue_dropped, not silently absorbed.
        self._ai_queue: deque = deque(maxlen=Config.AI_QUEUE_SIZE)
        self._ai_queue_lock = threading.Lock()
        self.ai_queue_dropped = 0

        self._capture_times: deque = deque(maxlen=90)

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._url_lock = threading.Lock()
        self._generation = 0  # bumped on every camera switch

        self.state = STATE_IDLE
        self.last_error: Optional[str] = None
        self.reconnect_attempt = 0
        self.reconnect_count = 0
        self.width = 0
        self.height = 0
        self.source_fps = 0.0
        self.frames_read = 0
        self.frames_dropped = 0
        self.connected_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, stream_url: str) -> None:
        """Start reading the given camera."""
        self._stop_event.clear()
        self.set_url(stream_url)

    def set_url(self, stream_url: str) -> None:
        """Point the capture at a different camera.

        A fresh reader thread is started for the new camera immediately. The
        previous thread owns its own capture and retires itself as soon as its
        (possibly blocking) read returns, so switching never waits on the
        network.
        """
        stream_url = (stream_url or '').strip()
        if not stream_url:
            return
        with self._url_lock:
            if stream_url == self._url:
                return
            self._url = stream_url
            self._generation += 1
            generation = self._generation
            self.reconnect_attempt = 0
            self.last_error = None
        # Drop the previous camera's frames so consumers never mix feeds.
        with self._frame_lock:
            self._frame = None
        with self._playout_lock:
            self._playout.clear()
        with self._ai_queue_lock:
            self._ai_queue.clear()
        self._capture_times.clear()
        self.width = self.height = 0
        self.source_fps = 0.0
        self.connected_at = None
        self.logger.info('Camera switched to %s', stream_url)
        self._spawn_reader(stream_url, generation)

    def _spawn_reader(self, url: str, generation: int) -> None:
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        thread = threading.Thread(
            target=self._reader_loop,
            args=(url, generation),
            name=f'streamer-{generation}',
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._url_lock:
            self._generation += 1  # retire every running reader
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout=3.0)
        self._threads.clear()
        self._set_state(STATE_STOPPED, 'Capture stopped')
        self.logger.info('Capture stopped.')

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[object]]:
        """Latest decoded frame, or (False, None) when nothing is buffered."""
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def read_latest(self, last_frame_id: int = -1):
        """Newest frame only when it differs from ``last_frame_id``.

        Returns ``(frame, frame_id)``; ``(None, last_frame_id)`` when no new
        frame has arrived, which lets the consumer skip redundant inference.
        This is what the AI thread uses: always the freshest frame, dropping
        anything captured while it was busy.
        """
        with self._frame_lock:
            if self._frame is None or self._frame_id == last_frame_id:
                return None, last_frame_id
            return self._frame.copy(), self._frame_id

    def pop_playout_frame(self) -> Optional[Tuple[object, float]]:
        """Oldest buffered frame plus its capture time, FIFO.

        This is what the render/preview thread drains at a steady pace. FIFO
        order (not "always latest") is what makes the *motion* look smooth:
        every captured frame eventually gets displayed, in the order the
        camera produced it, instead of skipping straight to whatever is
        newest and jumping over everything in between.
        """
        with self._playout_lock:
            if not self._playout:
                return None
            return self._playout.popleft()

    def playout_depth(self) -> int:
        with self._playout_lock:
            return len(self._playout)

    def clear_playout_buffer(self) -> None:
        with self._playout_lock:
            self._playout.clear()

    def pop_ai_frame(self) -> Optional[Tuple[object, float]]:
        """Oldest queued frame plus its capture time, FIFO.

        Lets the AI/tracker loop drain several distinct frames out of the
        same capture burst instead of only ever seeing whichever one was
        newest at the moment it happened to check (see Config.AI_QUEUE_SIZE).
        """
        with self._ai_queue_lock:
            if not self._ai_queue:
                return None
            return self._ai_queue.popleft()

    def ai_queue_depth(self) -> int:
        with self._ai_queue_lock:
            return len(self._ai_queue)

    def clear_ai_queue(self) -> None:
        with self._ai_queue_lock:
            self._ai_queue.clear()

    def is_active(self) -> bool:
        return self.state == STATE_ONLINE

    def get_stream_url(self) -> Optional[str]:
        with self._url_lock:
            return self._url

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f'{self.width}x{self.height}'
        return '-'

    @property
    def capture_fps(self) -> float:
        """Rolling actual frames-per-second the reader is delivering."""
        times = list(self._capture_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def stats(self) -> dict:
        return {
            'state': self.state,
            'stream_url': self.get_stream_url(),
            'resolution': self.resolution,
            'source_fps': round(self.source_fps, 2),
            'capture_fps': round(self.capture_fps, 2),
            'frames_read': self.frames_read,
            'frames_dropped': self.frames_dropped,
            'reconnect_attempt': self.reconnect_attempt,
            'reconnect_count': self.reconnect_count,
            'buffer_size': self.playout_depth(),
            'buffer_capacity': Config.PLAYOUT_BUFFER_FRAMES,
            'ai_backlog_frames': self.ai_queue_depth(),
            'ai_backlog_capacity': Config.AI_QUEUE_SIZE,
            'ai_backlog_dropped': self.ai_queue_dropped,
            'last_error': self.last_error,
            'uptime': round(time.time() - self.connected_at, 1) if self.connected_at else 0.0,
        }

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------
    def _reader_loop(self, url: str, generation: int) -> None:
        """Own one camera for as long as this generation is current."""
        while not self._stop_event.is_set() and not self._superseded(generation):
            capture = self._open(url, generation)
            if capture is None:
                continue  # _open already applied backoff / logged the attempt
            self._pump(capture, generation)
        self.logger.debug('Reader for generation %d retired.', generation)

    def _open(self, url: str, generation: int) -> Optional[cv2.VideoCapture]:
        """Open a capture once, with exponential backoff between attempts."""
        if self.reconnect_attempt == 0:
            self._set_state(STATE_CONNECTING, f'Connecting to {url}')
            self.logger.info('Camera connecting: %s', url)
        else:
            self._set_state(
                STATE_RECONNECTING,
                f'Reconnect attempt {self.reconnect_attempt}/{Config.RECONNECT_MAX_ATTEMPTS}',
            )
            self.logger.warning(
                'Reconnect attempt %d/%d for %s',
                self.reconnect_attempt, Config.RECONNECT_MAX_ATTEMPTS, url,
            )

        was_reconnecting = self.reconnect_attempt > 0
        capture = None
        try:
            if '.m3u8' in url.lower() or url.startswith(('http://', 'https://')):
                from hls_stream import CCTVStream
                capture = CCTVStream(url, width=1280, height=720).open()
            else:
                capture = cv2.VideoCapture(url)
            opened = capture.isOpened()
        except Exception as error:
            self.last_error = f'{type(error).__name__}: {error}'
            opened = False

        if self._superseded(generation):
            self._close(capture)
            return None

        if not opened:
            self._close(capture)
            self.last_error = self.last_error or 'VideoCapture could not open the stream'
            self._backoff(url)
            return None

        self.width = getattr(capture, 'width', 1280) or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.height = getattr(capture, 'height', 720) or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps = getattr(capture, 'source_fps', 20.0) or (capture.get(cv2.CAP_PROP_FPS) if hasattr(capture, 'get') else 20.0) or 20.0
        self.source_fps = fps if 0 < fps < 121 else 20.0
        self.reconnect_attempt = 0
        self.connected_at = time.time()
        self.last_error = None
        self._set_state(STATE_ONLINE, 'Camera connected')
        if was_reconnecting:
            self.reconnect_count += 1
            self.logger.info(
                'Camera recovered: %s [%s @ %.0f fps]', url, self.resolution, self.source_fps
            )
        else:
            self.logger.info(
                'Camera connected: %s [%s @ %.0f fps]', url, self.resolution, self.source_fps
            )
        return capture

    def _pump(self, capture: cv2.VideoCapture, generation: int) -> None:
        """Read frames until the stream dies or this generation is retired."""
        consecutive_failures = 0
        try:
            while not self._stop_event.is_set() and not self._superseded(generation):
                try:
                    ok, frame = capture.read()
                except Exception as error:
                    self.last_error = f'Read error: {error}'
                    ok, frame = False, None

                # A switch may have happened while read() was blocked.
                if self._superseded(generation):
                    return

                if not ok or frame is None or getattr(frame, 'size', 0) == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= Config.CAPTURE_READ_FAIL_LIMIT:
                        self.last_error = 'Stream stopped delivering frames'
                        self.logger.warning('Camera disconnected: %s', self.get_stream_url())
                        self._backoff(self.get_stream_url() or '')
                        return
                    # Brief pause; gaps between HLS segments are normal.
                    self._stop_event.wait(0.05)
                    continue

                consecutive_failures = 0
                self.frames_read += 1
                now = time.time()
                self._capture_times.append(now)
                with self._frame_lock:
                    if self._frame is not None:
                        # Previous frame was never consumed: drop it so the
                        # AI consumer always works on the newest image.
                        self.frames_dropped += 1
                    self._frame = frame
                    self._frame_id += 1
                    self._frame_time = now
                with self._playout_lock:
                    # Bounded deque: once full, appending silently drops the
                    # oldest buffered frame — the circular-buffer behaviour,
                    # sized to smooth over HLS segment-boundary stalls rather
                    # than to minimise latency (see Config.PLAYOUT_*).
                    self._playout.append((frame, now))
                with self._ai_queue_lock:
                    if len(self._ai_queue) >= self._ai_queue.maxlen:
                        self.ai_queue_dropped += 1
                        if self.ai_queue_dropped == 1 or self.ai_queue_dropped % 100 == 0:
                            self.logger.warning(
                                'AI backlog full (%d frames); dropping the oldest queued '
                                'frame. Processing is falling behind capture — this is a '
                                'safety valve against unbounded memory growth, not the '
                                'normal steady state (%d drop(s) so far).',
                                self._ai_queue.maxlen, self.ai_queue_dropped,
                            )
                    self._ai_queue.append((frame, now))
        finally:
            self._close(capture)

    def _backoff(self, url: str) -> None:
        """Sleep between reconnect attempts on a fixed schedule.

        Owns ``reconnect_attempt``: after this returns it holds the number of
        the attempt that is about to be made, so logs read 1/5, 2/5, ...
        Delays follow ``Config.RECONNECT_DELAYS`` (2s, 5s, 10s, then 30s for
        every attempt beyond that) instead of growing without bound, so a
        long outage settles into a predictable, low-noise retry rate.
        """
        self.reconnect_attempt += 1
        delays = Config.RECONNECT_DELAYS
        delay = delays[min(self.reconnect_attempt - 1, len(delays) - 1)]

        if self.reconnect_attempt == Config.RECONNECT_MAX_ATTEMPTS:
            self._set_state(
                STATE_FAILED,
                f'{Config.RECONNECT_MAX_ATTEMPTS} reconnect attempts failed; '
                f'retrying every {int(delays[-1])}s',
            )
            self.logger.error(
                'Camera unreachable after %d attempts (%s). Retrying every %ds. Last error: %s',
                Config.RECONNECT_MAX_ATTEMPTS, url, int(delays[-1]), self.last_error,
            )
        elif self.reconnect_attempt > Config.RECONNECT_MAX_ATTEMPTS:
            self.reconnect_attempt = Config.RECONNECT_MAX_ATTEMPTS  # stop the counter climbing forever
        self._stop_event.wait(delay)

    def _superseded(self, generation: int) -> bool:
        with self._url_lock:
            return generation != self._generation

    @staticmethod
    def _close(capture: Optional[cv2.VideoCapture]) -> None:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def _set_state(self, state: str, reason: str) -> None:
        if state == self.state:
            return
        self.state = state
        if self._on_state_change is not None:
            try:
                self._on_state_change(state, reason)
            except Exception as error:  # a listener must never kill capture
                self.logger.debug('State listener failed: %s', error)
