"""Detection pipeline.

Threading model
---------------
* ``Streamer`` (see services/streamer.py) owns the capture thread. It feeds
  three independent buffers on every successful capture: a single "latest
  frame" slot (unused by this module directly), a bounded playout ring
  buffer, and a bounded AI backlog queue.
* The **AI thread** here (``_ai_loop``) drains the AI backlog
  (``pop_ai_frame()``) strictly FIFO, running a full YOLO detect+track pass
  on every single frame it pops — see "Accuracy-first: zero intentional
  frame skip" below.
* The **render thread** (``_render_loop``) is separate and runs at its own
  steady pace (``Config.PLAYOUT_TARGET_FPS``), draining the playout buffer
  and drawing the *last known* detection boxes onto each frame before
  encoding it to JPEG. This is what keeps the live preview watchable even
  while the AI thread is working through a backlog: the preview does not
  wait on AI, it just shows the most recently published boxes on whatever
  frame is next in the playout buffer. The two threads only share the
  small, lock-guarded ``_render_state`` snapshot — never a live, mutable
  detection list.
* Database writes and snapshot encoding are handed to a small thread pool so
  neither the AI loop nor the render loop is ever blocked by disk or MySQL.

Accuracy-first: zero intentional frame skip
--------------------------------------------
This deployment's stated priority is detection/tracking accuracy over
latency, FPS, or CPU/GPU/memory cost, with frame arrival already irregular
by nature (an HLS crawl, not a fixed-rate RTSP feed) and some processing lag
explicitly accepted. A previous "YOLO/tracking-interpolation" design used to
live here: it ran a real YOLO pass on only every 3rd-5th captured frame and
filled the gaps with the tracker's own Kalman motion model — no new
detections at all on those frames — with a CPU-adaptive backoff and an idle
throttle that pushed the same trade even further under load or on an empty
road. All of that has been removed. ``_ai_loop`` now does one thing: pop the
next frame off the backlog queue (``Streamer.pop_ai_frame``, FIFO, fed by
the reader thread with every frame it captures) and run a real YOLO pass on
it (``VehicleTracker.detect_and_track``), in capture order, however long
that takes. There is no pacing sleep on this loop and no fallback to
interpolation — if YOLO is slower than the camera, the backlog simply grows
and the AI thread keeps working through it, exactly as "process frames
sequentially, queue if necessary, accuracy first, latency second" asks for.

This is not free: if sustained processing throughput stays below sustained
capture throughput for long enough (which is plausible under the heavier
accuracy-first inference settings — see config.py), the backlog queue will
keep growing. ``Config.AI_QUEUE_SIZE`` bounds it as a safety valve against
unbounded memory growth (Streamer logs loudly when it actually has to drop
the oldest queued frame — see ``Streamer.ai_queue_dropped``), but no queue
size changes the underlying arithmetic: a camera that is consistently busier
than the model can keep up with will run the AI side increasingly behind
real time. ``VehicleTracker.interpolate`` still exists (nothing about its
API changed) for anyone who wants the old latency-first behaviour back, it
is simply not called from this loop any more.

Every database access is wrapped in an explicit application context, which is
what makes writes from these background threads legal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional

import cv2

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from config import Config
from services.cctv_service import CCTVService
from services.detector import Detector
from services.preprocessor import FramePreprocessor
from services.roi import RegionOfInterest
from services.snapshot import SnapshotService
from services.speed_calculator import SpeedEstimator
from services.tracker import VehicleTracker
from services.vehicle_filter import VehicleFilter

logger = logging.getLogger(__name__)

STATUS_COLOURS = {
    'Overspeed': (0, 0, 255),
    'Normal': (0, 200, 0),
    'Measuring': (0, 200, 255),
}


class DetectionPipeline:
    """Owns the whole live-detection lifecycle."""

    def __init__(self, app: Any):
        self.app = app
        self.logger = logger

        self.roi = RegionOfInterest()
        self.cctv = CCTVService(source_url=app.config.get('CCTV_STREAM_URL'))
        self.detector = Detector(app.config.get('YOLO_WEIGHTS_PATH'))
        self.tracker = VehicleTracker(
            detector=self.detector,
            confidence=Config.DETECTION_CONFIDENCE,
            iou=Config.DETECTION_IOU,
        )
        self.preprocessor = FramePreprocessor()
        self.vehicle_filter = VehicleFilter()
        self.speed_estimator = SpeedEstimator(
            pixels_per_meter=app.config.get('PIXELS_PER_METER'),
            min_speed=app.config.get('MIN_SPEED'),
            max_speed=app.config.get('MAX_SPEED'),
            speed_limit=app.config.get('SPEED_LIMIT'),
            roi=self.roi,
        )
        self.snapshots = SnapshotService(app.config.get('SNAPSHOT_DIR'))

        # Latest rendered frame for the MJPEG endpoint. Only the render
        # thread ever writes this; ``_jpeg_ready`` wakes up any waiting
        # mjpeg_stream() generators the instant a new one is published,
        # instead of every viewer polling on its own timer and re-sending
        # duplicate bytes for the same encoded frame.
        self._jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()
        self._jpeg_ready = threading.Event()
        self._raw_frame = None
        self._raw_lock = threading.Lock()

        # Detection results computed by the AI thread, drawn by the render
        # thread. Holds plain snapshots (tuples/dicts), never the live
        # DetectionResult objects, so the render thread can never observe a
        # half-updated object while the AI thread is still mutating it.
        self._render_state: Dict[str, Any] = {'boxes': [], 'hud': []}
        self._render_lock = threading.Lock()

        self._running = False
        self._ai_thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._workers = ThreadPoolExecutor(max_workers=3, thread_name_prefix='pipeline-io')

        # Live statistics.
        self._stats_lock = threading.RLock()
        self._counted_tracks: set[int] = set()
        self._recorded_tracks: set[int] = set()
        self._counts: Dict[str, int] = defaultdict(int)
        self._overspeed_count = 0
        self._speed_sum = 0.0
        self._speed_n = 0
        self._ai_frame_times: deque = deque(maxlen=30)
        self._render_frame_times: deque = deque(maxlen=60)
        self._fps = 0.0
        self._render_fps = 0.0
        self._latency_ms = 0.0
        self._ai_processed = 0
        self._last_detections: List = []
        self._recent_events: deque = deque(maxlen=30)
        self._db_errors = 0
        self._db_saves = 0
        self.started_at = time.time()

        # Every AI-loop cycle now runs a real YOLO pass (accuracy-first, see
        # the module docstring) — these track its own throughput and, purely
        # for the requested benchmark/observability fields, how far the AI
        # side is currently lagging behind live capture. None of it feeds
        # back into whether a frame gets processed any more.
        self._frame_counter = 0
        self._yolo_frame_times: deque = deque(maxlen=20)
        self._yolo_inference_times: deque = deque(maxlen=20)
        self._yolo_fps = 0.0
        self._ai_lag_seconds = 0.0
        self._cpu_percent = 0.0
        self._cpu_checked_at = 0.0
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)  # prime the delta baseline
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        if not self.detector.is_ready:
            self.logger.error(
                'Pipeline starting without a model; the stream will show raw video only.'
            )
        self._running = True
        self._ai_thread = threading.Thread(target=self._ai_loop, name='pipeline-ai', daemon=True)
        self._ai_thread.start()
        self._render_thread = threading.Thread(
            target=self._render_loop, name='pipeline-render', daemon=True
        )
        self._render_thread.start()
        # Camera discovery touches the network: keep startup non-blocking.
        threading.Thread(target=self.cctv.start, name='cctv-start', daemon=True).start()
        self.logger.info('Detection pipeline started.')

    def stop(self) -> None:
        self._running = False
        if self._ai_thread is not None:
            self._ai_thread.join(timeout=5.0)
        if self._render_thread is not None:
            self._render_thread.join(timeout=5.0)
        self.cctv.stop()
        self._workers.shutdown(wait=False)
        self.logger.info('Detection pipeline stopped.')

    # ------------------------------------------------------------------
    # AI thread: strict FIFO backlog drain, every frame gets a real YOLO pass
    # ------------------------------------------------------------------
    def _ai_loop(self) -> None:
        """Accuracy-first: see the module docstring for the full rationale.

        No pacing sleep beyond a short idle wait when the backlog is empty,
        and no interpolation fallback — every popped frame gets a full
        ``detect_and_track`` call, in the order the camera captured it,
        regardless of how long that takes.
        """
        while self._running:
            popped = self.cctv.streamer.pop_ai_frame()
            if popped is None:
                time.sleep(0.02)
                continue
            frame, captured_at = popped

            tick_started = time.perf_counter()
            try:
                self._process_frame(frame, captured_at)
            except Exception as error:  # a bad frame must never kill the loop
                self.logger.exception('Frame processing failed: %s', error)

            elapsed = time.perf_counter() - tick_started
            with self._stats_lock:
                self._ai_processed += 1
                self._ai_frame_times.append(elapsed)
                total = sum(self._ai_frame_times)
                self._fps = (len(self._ai_frame_times) / total) if total > 0 else 0.0
                self._ai_lag_seconds = max(0.0, time.time() - captured_at)
            self._sample_cpu(time.time())

    def _process_frame(self, frame, captured_at: float) -> None:
        now = time.time()
        processed = self.preprocessor.process(frame)

        raw_detections = self.tracker.detect_and_track(processed)
        with self._stats_lock:
            self._yolo_frame_times.append(now)
            self._yolo_inference_times.append(self.tracker.last_inference_ms)
            total = self._yolo_frame_times[-1] - self._yolo_frame_times[0]
            self._yolo_fps = (
                (len(self._yolo_frame_times) - 1) / total
                if len(self._yolo_frame_times) >= 2 and total > 0 else 0.0
            )
        self._frame_counter += 1

        # Reject implausible boxes (too small, wrong shape, on the frame
        # edge, or not yet confirmed as a persisting object) before anything
        # downstream ever sees them. Uses the live-adjustable confidence
        # slider (self.tracker.confidence), same as before the accuracy-first
        # detour — see config.py's DETECTION_CONFIDENCE note for why a
        # separate, lower detector-only floor was tried and reverted.
        detections = self.vehicle_filter.filter(
            raw_detections, frame.shape, confidence_threshold=self.tracker.confidence
        )

        height, width = frame.shape[:2]
        # Vehicles outside the ROI are neither measured nor counted.
        for detection in detections:
            detection.in_roi = self.roi.contains(detection.ground_point, width, height)
        inside = [detection for detection in detections if detection.in_roi]

        self.speed_estimator.update(inside, frame.shape)
        self._update_stats(inside)
        self._handle_events(frame, inside)
        self._publish_render_state(detections)

        with self._stats_lock:
            self._last_detections = [self._detection_payload(d) for d in inside]

    def _sample_cpu(self, now: float) -> None:
        """Informational only: reported in stats/benchmarks, never throttles anything."""
        if psutil is None:
            return
        if now - self._cpu_checked_at < Config.YOLO_CPU_CHECK_INTERVAL_SECONDS:
            return
        self._cpu_checked_at = now
        try:
            self._cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def _publish_render_state(self, detections: List) -> None:
        """Snapshot plain values for the render thread to draw later.

        Only primitives cross the lock here (box coordinates, a colour
        tuple, a caption string) — never the ``DetectionResult`` objects
        themselves, which the AI thread keeps mutating on every cycle.
        """
        boxes = []
        for detection in detections:
            if not detection.in_roi:
                boxes.append({'box': list(detection.box), 'colour': (110, 110, 110), 'caption': None})
                continue
            colour = STATUS_COLOURS.get(detection.status, (0, 200, 0))
            speed = float(detection.speed or 0.0)
            if speed >= self.speed_estimator.min_speed:
                suffix = ' *' if detection.measured else ''
                caption = f'#{detection.track_id} {detection.label_id} {speed:.0f} km/h{suffix}'
            else:
                caption = f'#{detection.track_id} {detection.label_id} ...'
            boxes.append({'box': list(detection.box), 'colour': colour, 'caption': caption})

        stats = self.tracker.stats()
        camera = self.cctv.current_camera_name or 'no camera'
        backlog = self.cctv.streamer.ai_queue_depth()
        hud = [
            f'{camera}  |  {self.cctv.streamer.resolution}  |  {self.cctv.current_status}',
            f'{stats["model"]} / {stats["tracker"]}  |  '
            f'conf {stats["confidence"]:.2f} iou {stats["iou"]:.2f}  |  '
            f'{self.preprocessor.last_profile}',
            f'Capture {self.cctv.streamer.capture_fps:.0f} FPS  |  '
            f'YOLO {self._yolo_fps:.1f} FPS (every frame)  |  '
            f'Inference {stats["inference_ms"]:.0f} ms  |  '
            f'Backlog {backlog} frame(s)  |  '
            f'Lag {self._ai_lag_seconds:.1f}s  |  '
            f'Tracks {len(boxes)}',
        ]
        with self._render_lock:
            self._render_state = {'boxes': boxes, 'hud': hud}

    # ------------------------------------------------------------------
    # Render thread: fixed cadence, cheap draw + encode only, never AI
    # ------------------------------------------------------------------
    def _render_loop(self) -> None:
        target_interval = 1.0 / Config.PLAYOUT_TARGET_FPS
        prefill_deadline = time.time() + Config.PLAYOUT_PREFILL_TIMEOUT
        last_frame = None

        while self._running:
            tick_started = time.perf_counter()

            # Let the playout buffer build a small cushion before the first
            # frame, rather than draining it frame-by-frame as it trickles
            # in (which would just reproduce the raw stall pattern).
            if (
                self.cctv.streamer.playout_depth() < Config.PLAYOUT_PREFILL_FRAMES
                and time.time() < prefill_deadline
                and last_frame is None
            ):
                time.sleep(0.05)
                continue

            popped = self.cctv.streamer.pop_playout_frame()
            if popped is None:
                if last_frame is not None:
                    self._render_and_publish(last_frame)
                time.sleep(target_interval)
                continue

            frame, captured_at = popped
            last_frame = (frame, captured_at)
            with self._raw_lock:
                self._raw_frame = frame
            self._render_and_publish((frame, captured_at))

            with self._stats_lock:
                self._latency_ms = (time.time() - captured_at) * 1000.0
                self._render_frame_times.append(time.time())
                if len(self._render_frame_times) >= 2:
                    span = self._render_frame_times[-1] - self._render_frame_times[0]
                    self._render_fps = (
                        (len(self._render_frame_times) - 1) / span if span > 0 else 0.0
                    )

            elapsed = time.perf_counter() - tick_started
            remaining = target_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _render_and_publish(self, frame_data) -> None:
        # Copy so the retained ``last_frame`` is never mutated in place —
        # it may be redrawn again on a later tick if the playout buffer is
        # momentarily empty (e.g. right after a camera switch).
        frame, _ = frame_data
        overlay = frame.copy()
        self.roi.draw(overlay)
        with self._render_lock:
            boxes = list(self._render_state['boxes'])
            hud = list(self._render_state['hud'])
        for item in boxes:
            x1, y1, x2, y2 = item['box']
            if item['caption'] is None:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), item['colour'], 1)
                continue
            cv2.rectangle(overlay, (x1, y1), (x2, y2), item['colour'], 2)
            self._draw_label(overlay, item['caption'], x1, y1, item['colour'])
        self._draw_hud(overlay, hud)
        self._encode(overlay)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _update_stats(self, detections: List) -> None:
        with self._stats_lock:
            for detection in detections:
                track_id = detection.track_id
                if track_id is None or track_id in self._counted_tracks:
                    continue
                self._counted_tracks.add(track_id)
                self._counts[detection.class_name] += 1

    def _handle_events(self, frame, detections: List) -> None:
        """Persist confirmed measurements and capture overspeed snapshots."""
        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                continue
            with self._stats_lock:
                already = track_id in self._recorded_tracks
            if already or not self.speed_estimator.is_recordable(detection):
                continue

            with self._stats_lock:
                self._recorded_tracks.add(track_id)
                self._speed_sum += float(detection.speed)
                self._speed_n += 1
                if detection.status == 'Overspeed':
                    self._overspeed_count += 1
                self._recent_events.appendleft(self._detection_payload(detection))

            snapshot_path = None
            if self.snapshots.should_capture(track_id, detection.status == 'Overspeed'):
                snapshot_path = self.snapshots.save(
                    frame, track_id=track_id, box=detection.box
                )
            self._workers.submit(
                self._persist,
                track_id=track_id,
                vehicle_type=detection.label_id,
                speed=float(detection.speed),
                status=detection.status,
                confidence=float(detection.confidence),
                snapshot=snapshot_path,
            )

    def _persist(
        self,
        track_id: int,
        vehicle_type: str,
        speed: float,
        status: str,
        confidence: float,
        snapshot: Optional[str],
    ) -> None:
        """Write one detection. Failures are logged, never raised."""
        # Imported here so the module graph stays acyclic at import time.
        from db import db
        from models.detection import Detection

        started = time.perf_counter()
        try:
            with self.app.app_context():
                record = Detection(
                    vehicle_id=str(track_id),
                    track_id=int(track_id),
                    vehicle_type=vehicle_type,
                    speed=round(float(speed), 2),
                    status=status,
                    camera_name=self.cctv.current_camera_name,
                    stream_url=self.cctv.current_camera_url,
                    snapshot=snapshot,
                    confidence=round(float(confidence), 3),
                )
                db.session.add(record)
                db.session.commit()
                db.session.remove()
            with self._stats_lock:
                self._db_saves += 1
            self.logger.info(
                'Database Save: track=%s type=%s speed=%.1f km/h (%.0f ms)',
                track_id, vehicle_type, speed, (time.perf_counter() - started) * 1000,
            )
        except Exception as error:
            with self._stats_lock:
                self._db_errors += 1
            self.logger.error('Database save failed for track %s: %s', track_id, error)
            try:
                with self.app.app_context():
                    db.session.rollback()
                    db.session.remove()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Drawing primitives shared by the render loop
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_label(frame, text: str, x: int, y: int, colour) -> None:
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y - height - 8, 0)
        cv2.rectangle(frame, (x, top), (x + width + 8, top + height + 8), colour, -1)
        cv2.putText(
            frame, text, (x + 4, top + height + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_hud(frame, lines: List[str]) -> None:
        height = 24 * len(lines) + 10
        cv2.rectangle(frame, (0, 0), (frame.shape[1], height), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            cv2.putText(
                frame, line, (10, 22 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA,
            )

    def _encode(self, frame) -> None:
        try:
            ok, buffer = cv2.imencode(
                '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
            )
        except Exception as error:
            self.logger.debug('Encode failed: %s', error)
            return
        if not ok:
            return
        with self._jpeg_lock:
            self._jpeg = buffer.tobytes()
        self._jpeg_ready.set()

    # ------------------------------------------------------------------
    # Public API used by the routes
    # ------------------------------------------------------------------
    def mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Multipart JPEG stream for the dashboard's <img> element.

        Blocks on ``_jpeg_ready`` (bounded by a timeout, never forever) so
        each viewer is woken up exactly when the render thread publishes a
        new frame, rather than polling on its own timer and re-sending the
        same encoded bytes multiple times per real update. Encoding only
        ever happens in the render thread — this generator just serves
        whatever is already in memory.
        """
        boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        placeholder_sent = False
        while self._running:
            with self._jpeg_lock:
                frame = self._jpeg
            if frame is None:
                if not placeholder_sent:
                    placeholder = self._placeholder()
                    if placeholder is not None:
                        yield boundary + placeholder + b'\r\n'
                        placeholder_sent = True
                time.sleep(0.1)
                continue
            placeholder_sent = False
            yield boundary + frame + b'\r\n'
            self._jpeg_ready.wait(timeout=0.5)
            self._jpeg_ready.clear()

    def _placeholder(self) -> Optional[bytes]:
        """A 'connecting' card shown before the first frame arrives."""
        try:
            import numpy as np

            canvas = np.full((360, 640, 3), 18, dtype=np.uint8)
            message = self.cctv.current_status or 'CONNECTING'
            cv2.putText(
                canvas, 'Waiting for CCTV stream', (110, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, str(message), (110, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 190, 240), 1, cv2.LINE_AA,
            )
            ok, buffer = cv2.imencode('.jpg', canvas)
            return buffer.tobytes() if ok else None
        except Exception:
            return None

    def select_camera(self, camera_url: str) -> bool:
        """Switch camera and clear per-camera state."""
        if not self.cctv.select_camera(camera_url):
            return False
        self.tracker.reset()
        self.speed_estimator.reset()
        # Track ids restart on the new camera, so the streak-confirmation
        # state (and cumulative raw/valid/rejected counters) must too.
        self.vehicle_filter.reset()
        # select_camera() on CCTVService already clears the streamer's raw
        # frame; the playout buffer and AI queue are separate and need
        # clearing too, or the render/AI threads would briefly work through
        # the old camera's tail.
        self.cctv.streamer.clear_playout_buffer()
        self.cctv.streamer.clear_ai_queue()
        with self._jpeg_lock:
            self._jpeg = None
        with self._render_lock:
            self._render_state = {'boxes': [], 'hud': []}
        with self._stats_lock:
            self._last_detections = []
            self._render_frame_times.clear()
            self._render_fps = 0.0
            self._latency_ms = 0.0
            # Track ids restart per camera, so the "already seen" sets must go
            # too: otherwise a new camera reusing id 5 would be treated as
            # already counted and never recorded.
            self._counted_tracks.clear()
            self._recorded_tracks.clear()
        self.snapshots.prune(max_age=0.0)

        # New camera, new scene: every frame already gets a real YOLO pass,
        # so there is no scheduling state left to reset — just the rolling
        # throughput stats, so they reflect the new camera immediately
        # instead of averaging in numbers from the old one.
        self._frame_counter = 0
        self._yolo_frame_times.clear()
        self._yolo_inference_times.clear()
        self._yolo_fps = 0.0
        self._ai_lag_seconds = 0.0

        self.logger.info('Pipeline retargeted to %s', camera_url)
        return True

    def update_settings(
        self,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        preprocess: Optional[bool] = None,
        speed_limit: Optional[float] = None,
        pixels_per_meter: Optional[float] = None,
    ) -> dict:
        self.tracker.configure(confidence=confidence, iou=iou)
        self.preprocessor.configure(enabled=preprocess)
        self.speed_estimator.configure(
            pixels_per_meter=pixels_per_meter, speed_limit=speed_limit
        )
        return self.settings()

    def settings(self) -> dict:
        return {
            'confidence': round(self.tracker.confidence, 2),
            'iou': round(self.tracker.iou, 2),
            'preprocess': self.preprocessor.enabled,
            'speed_limit': self.speed_estimator.speed_limit,
            'pixels_per_meter': self.speed_estimator.pixels_per_meter,
            'min_speed': self.speed_estimator.min_speed,
            'max_speed': self.speed_estimator.max_speed,
        }

    def manual_capture(self) -> Optional[str]:
        """Save the current frame on operator request."""
        with self._raw_lock:
            frame = None if self._raw_frame is None else self._raw_frame.copy()
        if frame is None:
            return None
        return self.snapshots.save(frame, prefix='manual')

    def live_stats(self) -> dict:
        stream_stats = self.cctv.streamer.stats()
        with self._stats_lock:
            counts = dict(self._counts)
            total = sum(counts.values())
            average = (self._speed_sum / self._speed_n) if self._speed_n else 0.0
            ai_dropped = max(0, stream_stats['frames_read'] - self._ai_processed)
            payload = {
                'total': total,
                'car': counts.get('car', 0),
                'motorcycle': counts.get('motorcycle', 0),
                'bus': counts.get('bus', 0),
                'truck': counts.get('truck', 0),
                'overspeed': self._overspeed_count,
                'average_speed': round(average, 2),
                'measured': self._speed_n,
                # 'fps' kept for backward compatibility with the dashboard;
                # it is the AI/inference loop's own cadence.
                'fps': round(self._fps, 1),
                'inference_fps': round(self._fps, 1),
                'render_fps': round(self._render_fps, 1),
                'capture_fps': stream_stats['capture_fps'],
                'latency_ms': round(self._latency_ms, 0),
                'dropped_frames': stream_stats['frames_dropped'],
                'ai_dropped_frames': ai_dropped,
                'buffer_size': stream_stats['buffer_size'],
                'buffer_capacity': stream_stats['buffer_capacity'],
                'reconnect_count': stream_stats['reconnect_count'],
                'active_tracks': len(self._last_detections),
                'db_saves': self._db_saves,
                'db_errors': self._db_errors,
                'uptime': round(time.time() - self.started_at, 1),
                'detections': list(self._last_detections),
                'events': list(self._recent_events)[:10],
                'ai_backlog_frames': stream_stats['ai_backlog_frames'],
                'ai_backlog_capacity': stream_stats['ai_backlog_capacity'],
                'ai_backlog_dropped': stream_stats['ai_backlog_dropped'],
            }
        payload.update(self.tracker.stats())
        payload['preprocess'] = self.preprocessor.stats()
        payload['tracked'] = self.speed_estimator.tracked_count
        payload['crossing'] = self.speed_estimator.crossing_stats()
        payload['vehicle_filter'] = self.vehicle_filter.stats()

        # Accuracy-first throughput stats (see the module docstring): every
        # cycle runs a real YOLO pass now, so yolo_fps and the AI loop's own
        # fps converge — both are still reported since the dashboard/older
        # tooling may read either name. ai_lag_seconds is the new one worth
        # watching: how far behind live capture the AI side currently is.
        with self._stats_lock:
            avg_inference_ms = (
                sum(self._yolo_inference_times) / len(self._yolo_inference_times)
                if self._yolo_inference_times else 0.0
            )
            payload['tracker_fps'] = round(self._fps, 1)
            payload['yolo_fps'] = round(self._yolo_fps, 1)
            payload['yolo_avg_inference_ms'] = round(avg_inference_ms, 1)
            payload['ai_lag_seconds'] = round(self._ai_lag_seconds, 2)
            payload['cpu_percent'] = round(self._cpu_percent, 1)
        return payload

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._counted_tracks.clear()
            self._recorded_tracks.clear()
            self._counts.clear()
            self._overspeed_count = 0
            self._speed_sum = 0.0
            self._speed_n = 0
            self._recent_events.clear()
        self.logger.info('Live statistics reset.')

    @staticmethod
    def _detection_payload(detection) -> dict:
        return {
            'track_id': detection.track_id,
            'vehicle_type': detection.label_id,
            'class_name': detection.class_name,
            'speed': round(float(detection.speed or 0.0), 1),
            'status': detection.status,
            'confidence': round(float(detection.confidence), 2),
            'measured': bool(detection.measured),
            'box': list(detection.box),
        }

    @property
    def is_running(self) -> bool:
        return self._running
