"""Application configuration.

Only the variables documented in ``.env`` are read from the environment.
Every other tunable lives here as a plain constant so the deployment
contract (the ``.env`` format) never has to grow.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment without ever raising."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'on'}


def _resolve_path(raw: str, fallback: Path) -> Path:
    """Resolve a possibly relative env path against the project root."""
    if not raw or not str(raw).strip():
        return fallback
    candidate = Path(str(raw).strip())
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def build_database_uri() -> str:
    """Database URI taken verbatim from DATABASE_URL, with a safe default."""
    env_url = (os.environ.get('DATABASE_URL') or '').strip()
    if env_url:
        return env_url
    return 'mysql+pymysql://root:@127.0.0.1:3306/speed_tracker'


class Config:
    # ------------------------------------------------------------------
    # Values sourced from .env
    # ------------------------------------------------------------------
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = _env_int('PORT', 8000)
    FLASK_DEBUG = _env_bool('FLASK_DEBUG', False)

    SQLALCHEMY_DATABASE_URI = build_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {'pool_pre_ping': True, 'pool_recycle': 900}

    YOLO_WEIGHTS_PATH = str(_resolve_path(
        os.environ.get('YOLO_WEIGHTS_PATH', ''), BASE_DIR / 'weights' / 'yolo26m.pt'
    ))

    # Entry point used for crawling when AUTO_FETCH_CAMERAS is enabled,
    # and as the direct fallback stream when crawling yields nothing.
    CCTV_STREAM_URL = os.environ.get('CCTV_STREAM_URL', '').strip()

    PIXELS_PER_METER = _env_float('PIXELS_PER_METER', 8.0)
    SPEED_LIMIT = _env_float('SPEED_LIMIT', 60.0)
    MIN_SPEED = _env_float('MIN_SPEED', 5.0)
    MAX_SPEED = _env_float('MAX_SPEED', 240.0)

    SNAPSHOT_DIR = str(_resolve_path(
        os.environ.get('SNAPSHOT_DIR', ''), BASE_DIR / 'static' / 'snapshot'
    ))
    LOG_DIR = str(_resolve_path(os.environ.get('LOG_DIR', ''), BASE_DIR / 'logs'))

    SECRET_KEY = 'speed-tracker'

    # ------------------------------------------------------------------
    # Fixed application constants (deliberately not env driven)
    # ------------------------------------------------------------------
    LOG_LEVEL = 'INFO'

    # Camera discovery
    AUTO_FETCH_CAMERAS = True
    CCTV_CRAWL_TIMEOUT = 20
    CCTV_PROBE_TIMEOUT = 8
    CCTV_MAX_PROBE_WORKERS = 12

    # Capture / reconnect behaviour.
    # Delay before each successive reconnect attempt; the last value repeats
    # for every attempt beyond the list (2s, 5s, 10s, then every 30s).
    RECONNECT_DELAYS = (2.0, 5.0, 10.0, 30.0)
    RECONNECT_MAX_ATTEMPTS = 5  # attempts before the state escalates to FAILED
    CAPTURE_OPEN_TIMEOUT = 20.0
    CAPTURE_READ_FAIL_LIMIT = 30

    # Playout (preview) jitter buffer.
    #
    # Audited against the live HLS source: capture.read() delivers frames in
    # bursts (a whole segment decodes in a few hundred ms) separated by
    # stalls up to ~2.6s while the next segment is fetched over HTTP — this
    # happens inside FFmpeg's own blocking read, so no amount of OpenCV-level
    # buffer/option tuning removes it. Simulated against real capture timing
    # from two cameras: a buffer smaller than ~35-40 frames (~1.75-2s) still
    # shows the display holding a stale frame while waiting on the next
    # segment; 45 frames (~2.2s @ ~20fps) with a steady 12 FPS playout showed
    # zero stalls on both. This trades ~2.2s of extra preview latency for a
    # feed that never visibly freezes. Detection/tracking are unaffected —
    # the AI loop still always reads the single freshest captured frame,
    # bypassing this buffer entirely.
    PLAYOUT_BUFFER_FRAMES = 45
    PLAYOUT_TARGET_FPS = 12
    PLAYOUT_PREFILL_FRAMES = 24
    PLAYOUT_PREFILL_TIMEOUT = 2.5

    # AI-side backlog queue. Separate from the playout buffer above (which
    # feeds the render loop) — this one feeds the AI/tracker loop.
    #
    # Accuracy-first mode (see services/pipeline.py module docstring): the
    # reader thread enqueues every captured frame and the AI loop drains it
    # strictly FIFO, never skipping a frame to keep up with real time. That
    # means this is no longer a small "burst absorber" — it is the whole
    # backlog. Sized generously (minutes of frames at typical capture rates)
    # as a safety valve against unbounded memory growth, NOT as a target:
    # if sustained capture rate keeps exceeding sustained processing rate for
    # hours, no finite queue size prevents the backlog from growing — that is
    # arithmetic, not a bug. When this fills, the oldest queued frame is
    # dropped (logged loudly, see Streamer.ai_queue_dropped) rather than
    # silently discarded, so a persistently-full queue is visible instead of
    # a mystery. Easily lowered if the deployment machine has less memory
    # headroom than assumed here.
    AI_QUEUE_SIZE = 600

    # Model auto-selection, highest priority first.
    MODEL_PRIORITY = (
        'weights/yolo26m.pt',
        'weights/yolo11m.pt',
        'weights/yolov8m.pt',
        'weights/yolo26n.pt',
        'weights/yolo11n.pt',
        'weights/yolov8n.pt',
    )

    # COCO ids for the only classes we care about.
    VEHICLE_CLASS_IDS = (2, 3, 5, 7)
    VEHICLE_CLASS_LABELS = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    VEHICLE_CLASS_LABELS_ID = {2: 'Mobil', 3: 'Motor', 5: 'Bus', 7: 'Truk'}

    # Inference defaults for the yolo26m model.
    #
    # REGRESSION HISTORY — read before touching these again. An
    # "accuracy-first" pass briefly raised imgsz to 1536, dropped conf to
    # 0.10 and turned on ReID, reasoning that more pixels + a lower floor +
    # appearance matching would only ever help recall. In isolation (a
    # single-purpose benchmark script with nothing else running) that looked
    # true: more raw boxes, more small/distant boxes. In the real running
    # app it was a regression, confirmed against live telemetry, not
    # assumption:
    #   - imgsz alone: 589ms/call (832) -> 1649ms/call (1536), matching a
    #     real-footage sweep (21.1 -> 44.5 boxes/frame, ~2.8x cost for it).
    #   - conf 0.25->0.10 roughly tripled raw boxes/frame (17 -> 39-49);
    #     every extra box then paid a second ReID forward pass.
    #   - Combined, on the live server (real capture/render/DB threads all
    #     sharing the same CPU, not the isolated benchmark's clean run):
    #     inference climbed to 2000-2400ms typically, 8000+ms under worse
    #     contention, FPS collapsed 1.3 -> 0.4-0.5.
    #   - That slower cadence then broke MIN_CONFIRM_FRAMES (see below),
    #     which is the actual reason vehicles started going uncounted —
    #     raw YOLO detections were *up*, not down; they just stopped
    #     surviving the temporal-confirmation gate once real cycles were
    #     seconds apart instead of a fraction of a second apart.
    # All three (imgsz, conf, with_reid) were reverted to the values proven
    # to work well in production (with_reid stays off — trackers/
    # botsort_cctv.yaml). imgsz/conf were then RE-TUNED, this time
    # correctly: measured against real night footage from this deployment
    # (a glare-heavy urban intersection) with manually fixed ground-truth
    # frames (visually inspected, not assumed), rather than an isolated
    # single-purpose benchmark loop.
    #
    #   imgsz sweep (conf=0.25, same 7 real frames): 832 -> 32 raw boxes
    #   total, 1024 -> 40, 1280 -> 36, 1536 -> 38. 1024 wins, and not just
    #   in aggregate: on the frame with two clearly-visible motorcycles,
    #   832/1280/1536 each caught only one of the two, 1024 caught both —
    #   confirmed by viewing the annotated detections directly, not just
    #   counting boxes. Cost: 737ms -> 955ms/call, both far below the
    #   1536 profile's 1977ms that caused the earlier regression.
    #
    #   conf sweep (imgsz=1024): 0.25 -> 40 boxes, 0.15 -> 69, 0.10 -> 95,
    #   0.05 -> 161. 0.05 was inspected visually and is where real
    #   detections stop increasing and noise starts: duplicate boxes
    #   splitting a single already-detected car, and boxes on bare
    #   reflections/lights with no vehicle at all. 0.10 is the point right
    #   before that — every additional box at 0.10 checked corresponded to
    #   a real vehicle (most importantly, both motorcycles above were only
    #   found at conf<=0.16, invisible at 0.25).
    #
    #   iou (0.45/0.55/0.65 at imgsz=1024, conf=0.10): zero effect on box
    #   count on any of the 7 frames tested — left at 0.45.
    #
    # This is a different, smaller change than the reverted 1536/ReID
    # attempt (1024 vs 1536, no ReID) and was benchmarked the way that
    # attempt should have been: against the live camera's actual hard
    # conditions (night, glare), with real per-vehicle visual verification,
    # not just an aggregate box count.
    DETECTION_CONFIDENCE = 0.10
    DETECTION_IOU = 0.45
    INFERENCE_IMAGE_SIZE = 1024
    # Per-class NMS: an overlapping car and truck must never suppress each
    # other just because IoU is high — only same-class overlaps are merged.
    # Matches the ultralytics default; made explicit rather than changing
    # behaviour.
    DETECTION_AGNOSTIC_NMS = False
    # Ultralytics default; generous enough that no plausible single CCTV
    # frame's vehicle count would ever hit it. Also a no-op vs. leaving it
    # unset — made explicit for clarity, not a behaviour change.
    DETECTION_MAX_DET = 300
    # Test-time augmentation (augment=True) was evaluated and is NOT used:
    # this yolo26m export does not support it — ultralytics logs "Model does
    # not support 'augment=True', reverting to single-scale prediction" and
    # runs a normal single-scale pass anyway, so enabling it would only add
    # log spam with zero effect. Not set; augment stays at the ultralytics
    # default (False) in every inference call.

    # Frame handling: zero intentional skip (services/pipeline.py,
    # services/streamer.py).
    #
    # The previous "YOLO/tracking-interpolation" design ran a real YOLO pass
    # on only 1 in every 3-5 captured frames and filled the gaps with the
    # tracker's own Kalman motion model (no new detections at all) — a
    # vehicle that appeared and left within a gap was simply never seen.
    # That scheduler is gone; every frame from the reader thread is queued
    # (Config.AI_QUEUE_SIZE) and run through a real YOLO pass, in capture
    # order. This part of the redesign is NOT what caused the regression
    # documented above and is kept: with imgsz/conf/ReID reverted to their
    # proven-good values, a real cycle costs ~500-800ms again, so this loop
    # runs at roughly the same real-detection rate the old interval design
    # achieved (~1.3-1.8 calls/s) — except now *every* cycle is a genuine
    # detection instead of only ~1 in 3-5, which is strictly more accurate
    # at the same effective cost. See the module docstring in
    # services/pipeline.py for the full threading model and the documented
    # (now rarely exercised, given restored throughput) backlog-overflow
    # safety valve.
    #
    # VehicleTracker.interpolate() (services/tracker.py) is kept as a public
    # method — not deleted, not called from the main loop any more — so this
    # constant stays defined purely to keep that method's existing contract
    # working if anything still calls it directly.
    YOLO_MAX_INTERPOLATION_STALENESS = 6

    # How often the AI loop re-samples system-wide CPU%. Purely informational
    # now (surfaced in live_stats()/benchmarks) — nothing throttles on it any
    # more, see the frame-handling note above.
    YOLO_CPU_CHECK_INTERVAL_SECONDS = 2.0

    # Post-detection validity filter (services/vehicle_filter.py).
    #
    # Runs after the tracker, before speed estimation/counting/persistence.
    # Its job is to reject boxes that are still implausible as real vehicles
    # even after clearing DETECTION_CONFIDENCE above: too small to be
    # anything but noise, the wrong shape for a car/motorcycle/bus/truck,
    # sitting on the frame boundary (a common source of partial-object false
    # positives), or not yet confirmed as a persisting object.
    #
    # Geometry floors kept at the Phase-6 values and re-checked directly
    # against the same real night frames used to pick imgsz/conf above: at
    # imgsz=1024/conf=0.10, genuinely tiny noise/artifact boxes measured
    # 300-570px^2 while both real motorcycles measured 5200-8000px^2 — the
    # 500px^2 / 20px floor sits cleanly between them on this footage, so it
    # was left as-is rather than loosened blind.
    MIN_BBOX_WIDTH = 20
    MIN_BBOX_HEIGHT = 20
    MIN_BBOX_AREA = 500
    MIN_ASPECT_RATIO = 0.4
    MAX_ASPECT_RATIO = 4.5
    # A track must be seen this many times before counting as a confirmed
    # vehicle (anti-flicker). services/vehicle_filter.py counts this
    # cumulatively within VEHICLE_FILTER_CONFIRM_EXPIRY_SECONDS of wall-clock
    # time, NOT as strictly-consecutive processing cycles (see that
    # constant's note — this was a real, previously-shipped regression,
    # not a hypothetical).
    #
    # Set to 1 (a single real YOLO detection is enough to count) per this
    # deployment's explicit, repeated priority: recall over everything else,
    # including tolerating a slower/choppier feed. At imgsz=1024/conf=0.10
    # a real cycle costs ~950-1000ms; requiring even 2 hits risks losing a
    # vehicle that only crosses the frame in 1-2s no matter how generous the
    # expiry window is, since there may simply never be a second real
    # detection of it. The trade-off is explicit and accepted here: a lone
    # spurious detection now counts immediately too, instead of needing a
    # second occurrence to rule out a one-off flicker.
    MIN_CONFIRM_FRAMES = 1
    # A track's confirmation progress expires (resets to zero) if it goes
    # unseen for longer than this, instead of resetting on any single missed
    # cycle. A few real cycles' worth of grace at the ~500-800ms/call target
    # above, without keeping stale/gone tracks "half-confirmed" forever.
    VEHICLE_FILTER_CONFIRM_EXPIRY_SECONDS = 3.0
    # Fraction of the frame width/height treated as "touching the edge".
    # Left/right/top only — vehicles legitimately get cut off at the bottom
    # edge when very close to the camera, so that edge is not filtered.
    # Reverted to the Phase-6 values for the same reason as the size floors
    # above (the narrower margins were justified by the now-reverted imgsz).
    EDGE_MARGIN_LEFT = 0.05
    EDGE_MARGIN_RIGHT = 0.05
    EDGE_MARGIN_TOP = 0.03

    # Tracker preference, best first. The bundled CCTV profiles tune
    # thresholds for this deployment's frame rate — see
    # trackers/botsort_cctv.yaml for the full per-parameter rationale, and
    # its regression note re: ReID (tried, reverted). Stock configs are kept
    # as fallbacks in case a build rejects the custom file.
    TRACKER_PRIORITY = (
        'trackers/botsort_cctv.yaml',
        'trackers/bytetrack_cctv.yaml',
        'botsort.yaml',
        'bytetrack.yaml',
    )

    # Speed estimation
    SPEED_HISTORY_SIZE = 12
    SPEED_MIN_SAMPLES = 3
    PERSPECTIVE_TOP_SCALE = 0.35
    PERSPECTIVE_BOTTOM_SCALE = 1.0

    # Preprocessing
    PREPROCESS_ENABLED = True
    NIGHT_BRIGHTNESS_THRESHOLD = 80.0
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_GRID = 8

    # Frame plumbing
    FRAME_QUEUE_SIZE = 1
    JPEG_QUALITY = 80
    STATS_EMIT_INTERVAL = 1.0

    # Housekeeping
    TRACK_EXPIRY_SECONDS = 15.0
    SNAPSHOT_COOLDOWN_SECONDS = 5.0
