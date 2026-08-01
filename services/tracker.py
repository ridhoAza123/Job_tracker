"""Multi-object tracking on top of the YOLO model.

Owns a standalone BoTSORT/ByteTrack instance directly (constructed from the
same ``trackers/*.yaml`` configs used before) instead of going through
ultralytics' ``model.track()`` convenience wrapper. That single change is
what makes tracking-interpolation possible: ``model.track()`` always runs a
full YOLO forward pass internally, so there is no way to ask it for "just
the tracker's next predicted position, no new detections" — but the
standalone tracker object exposes exactly that, because its Kalman
predict/associate step runs every time ``.update()`` is called regardless of
whether any detections are passed in (empirically verified: feeding zero
detections still advances every existing track's predicted position, it
just isn't included in the *public* return value — see ``interpolate()``).

Two entry points reflect the two kinds of cycle the pipeline schedules:

* :meth:`detect_and_track` — a real YOLO pass feeds fresh detections into the
  tracker. Used on interval frames.
* :meth:`interpolate` — no YOLO at all; only the tracker's own motion model
  advances existing tracks. Used on the frames in between, and is roughly an
  order of magnitude cheaper than a YOLO pass on this hardware.

Both return the exact same ``DetectionResult`` shape, so nothing downstream
(vehicle_filter, speed_calculator, the render loop) needs to know or care
which kind of cycle produced a given box.

Global Motion Compensation is disabled in ``trackers/botsort_cctv.yaml``
(``gmc_method: none``): every camera here is fixed CCTV, so there is no
camera-induced motion to compensate for, and BoTSORT's optical-flow GMC step
compares the current frame against a cached previous one — a silent
resolution change between calls (e.g. a stream reconnect renegotiating a
different HLS rendition) made it throw a SparsePyrLKOpticalFlow assertion
every time. ``_guard_frame_shape`` below is the belt-and-suspenders half of
that fix: it forces a full tracker reset (Kalman tracks included, not just
GMC) the moment a frame's dimensions differ from the last one seen, since a
pixel-coordinate track from one resolution is meaningless at another
regardless of GMC.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from config import BASE_DIR, Config
from services.detector import DetectionResult, Detector, class_label, class_label_id

logger = logging.getLogger(__name__)


class VehicleTracker:
    """Detection + tracking, with automatic tracker selection."""

    def __init__(
        self,
        detector: Optional[Detector] = None,
        weights_path: Optional[str] = None,
        tracker_config: Optional[str] = None,
        classes: Sequence[int] = Config.VEHICLE_CLASS_IDS,
        confidence: float = Config.DETECTION_CONFIDENCE,
        iou: float = Config.DETECTION_IOU,
    ):
        self.logger = logger
        self.detector = detector or Detector(weights_path)
        self.classes = list(classes)
        self.confidence = float(confidence)
        self.iou = float(iou)

        self.tracker_config: Optional[str] = None
        self.tracker_name: str = 'none'
        self._tracker_instance: Optional[Any] = None
        self._tracking_supported = True
        self._lock = threading.Lock()

        self.last_inference_ms = 0.0  # cost of the last real YOLO+track cycle
        self.last_tracker_ms = 0.0    # cost of the last interpolation-only cycle
        self.last_track_count = 0

        # Every incoming frame must be the same size the tracker was last
        # fed — a silent resolution change (e.g. a stream reconnect that
        # renegotiates a different HLS rendition) would otherwise hand a
        # differently-shaped frame to internal state that assumes constant
        # dimensions, and every existing Kalman-filter track is in *pixel*
        # coordinates that stop meaning anything once the frame size shifts
        # anyway. Checked on every call; see _guard_frame_shape().
        self._expected_shape: Optional[tuple[int, int]] = None
        self.resolution_reset_count = 0

        self._select_tracker(tracker_config)

    # ------------------------------------------------------------------
    def _select_tracker(self, preferred: Optional[str] = None) -> None:
        """Construct the best standalone tracker this ultralytics build supports."""
        if not self.detector.is_ready:
            self.logger.error('Tracker disabled: no model loaded.')
            self._tracking_supported = False
            return

        candidates: List[str] = []
        if preferred:
            candidates.append(preferred)
        candidates.extend(
            entry for entry in Config.TRACKER_PRIORITY if entry not in candidates
        )
        candidates = [self._resolve_config(entry) for entry in candidates]

        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        probe_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        probe_boxes = Boxes(np.zeros((0, 6), dtype=np.float32), orig_shape=(64, 64)).cpu().numpy()

        for candidate in candidates:
            try:
                resolved_yaml = check_yaml(candidate)
                cfg = IterableSimpleNamespace(**YAML.load(resolved_yaml))
                cfg.device = 'cpu'
                tracker_cls = TRACKER_MAP.get(cfg.tracker_type)
                if tracker_cls is None:
                    raise ValueError(f'Unsupported tracker_type: {cfg.tracker_type}')

                # Dry-run on a throwaway instance, not the one we'll actually
                # use: STrack.activate() only auto-confirms a brand new track
                # on the tracker's very first frame_id. Probing with the
                # real instance would burn that frame_id on an empty test
                # frame, delaying confirmation of the first genuine vehicle.
                tracker_cls(args=cfg).update(probe_boxes, probe_frame)
                instance = tracker_cls(args=cfg)
                self._tracker_instance = instance
                self.tracker_config = candidate
                self.tracker_name = Path(candidate).stem.upper()
                self.logger.info('Tracker ready: %s', self.tracker_name)
                return
            except Exception as error:
                self.logger.warning(
                    'Tracker %s unavailable (%s). Trying the next one.', candidate, error
                )
        self.logger.error('No tracker available; falling back to plain detection.')
        self._tracking_supported = False
        self.tracker_name = 'DETECTION ONLY'

    @staticmethod
    def _resolve_config(entry: str) -> str:
        """Turn a project-relative tracker config into an absolute path.

        Bare names such as ``botsort.yaml`` are left untouched so ultralytics
        resolves them from its own bundled configs.
        """
        candidate = Path(entry)
        if candidate.is_absolute():
            return str(candidate)
        if len(candidate.parts) == 1:
            return entry  # ultralytics built-in
        local = BASE_DIR / candidate
        return str(local) if local.exists() else entry

    # ------------------------------------------------------------------
    def configure(
        self, confidence: Optional[float] = None, iou: Optional[float] = None
    ) -> None:
        """Apply live slider changes; takes effect on the next frame."""
        if confidence is not None:
            self.confidence = float(min(max(confidence, 0.05), 0.95))
        if iou is not None:
            self.iou = float(min(max(iou, 0.05), 0.95))

    def reset(self) -> None:
        """Drop tracker state, e.g. after switching camera."""
        if self._tracker_instance is not None:
            try:
                self._tracker_instance.reset()
            except Exception as error:
                self.logger.debug('Tracker reset failed, leaving state as-is: %s', error)
        self._expected_shape = None

    def _guard_frame_shape(self, frame: Any) -> None:
        """Force a full reset if the frame size changed since the last call.

        Covers reconnects and anything else that changes resolution without
        going through an explicit camera switch (which already calls
        :meth:`reset`) — a plain stream reconnect is handled entirely inside
        the Streamer's reader thread and never notifies this class directly,
        so this is the one place that reliably catches it regardless of
        cause.
        """
        shape = (frame.shape[0], frame.shape[1])
        if self._expected_shape is None:
            self._expected_shape = shape
            return
        if shape != self._expected_shape:
            self.logger.warning(
                'Frame size changed (%s -> %s); resetting tracker state.',
                self._expected_shape, shape,
            )
            self.reset()
            self._expected_shape = shape
            self.resolution_reset_count += 1

    # ------------------------------------------------------------------
    # Real YOLO cycle: fresh detections drive the tracker's association step
    # ------------------------------------------------------------------
    def detect_and_track(self, frame: Any) -> List[DetectionResult]:
        """Run YOLO and feed its output into the tracker. Never raises."""
        if frame is None or not self.detector.is_ready:
            return []
        if not self._tracking_supported or self._tracker_instance is None:
            return self._detect_only(frame)

        started = time.perf_counter()
        with self._lock:  # the tracker instance is not thread safe
            self._guard_frame_shape(frame)
            try:
                results = self.detector.model(
                    frame,
                    imgsz=Config.INFERENCE_IMAGE_SIZE,
                    conf=self.confidence,
                    iou=self.iou,
                    classes=self.classes,
                    agnostic_nms=Config.DETECTION_AGNOSTIC_NMS,
                    max_det=Config.DETECTION_MAX_DET,
                    verbose=False,
                )
                det = results[0].boxes.cpu().numpy()
                self._tracker_instance.update(det, frame)
                detections = self._extract_active_tracks(max_staleness=0)
            except Exception as error:
                self.logger.error('YOLO detect+track failed (%s); using plain detection.', error)
                detections = self._detect_only(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_inference_ms = round(elapsed_ms, 1)
        self.last_track_count = len(detections)
        return detections

    def _detect_only(self, frame: Any) -> List[DetectionResult]:
        """Untracked fallback for when no tracker could be constructed at all."""
        return self.detector.detect(
            frame, confidence=self.confidence, iou=self.iou, classes=self.classes
        )

    # ------------------------------------------------------------------
    # Skip cycle: no YOLO, just the tracker's own motion model
    # ------------------------------------------------------------------
    def interpolate(self, frame: Any) -> List[DetectionResult]:
        """Advance existing tracks with no new detections. Never raises.

        Costs roughly an order of magnitude less than :meth:`detect_and_track`
        on this hardware (~25-30ms vs ~300-380ms for yolo26m, measured against
        the live feed) since it skips the YOLO forward pass entirely.
        """
        if frame is None or not self.detector.is_ready:
            return []
        if not self._tracking_supported or self._tracker_instance is None:
            # Nothing to interpolate without a tracker; this is a rare,
            # already-degraded fallback path, not the normal one.
            return []

        started = time.perf_counter()
        with self._lock:
            self._guard_frame_shape(frame)
            try:
                from ultralytics.engine.results import Boxes

                empty = Boxes(
                    np.zeros((0, 6), dtype=np.float32), orig_shape=frame.shape[:2]
                ).cpu().numpy()
                self._tracker_instance.update(empty, frame)
                detections = self._extract_active_tracks(
                    max_staleness=Config.YOLO_MAX_INTERPOLATION_STALENESS
                )
            except Exception as error:
                self.logger.error('Tracker interpolation failed: %s', error)
                detections = []
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_tracker_ms = round(elapsed_ms, 1)
        self.last_track_count = len(detections)
        return detections

    # ------------------------------------------------------------------
    def _extract_active_tracks(self, max_staleness: int) -> List[DetectionResult]:
        """Read tracks directly off the tracker's own pools.

        Used after *both* a real update and an interpolation-only update —
        the tracker's public return value only reflects boxes matched to a
        detection in that exact call, but a track's ``frame_id`` freezes at
        its last real match while ``tracker.frame_id`` keeps advancing, so
        ``tracker.frame_id - track.frame_id`` is a clean staleness measure
        regardless of which kind of cycle produced it.
        """
        tracker = self._tracker_instance
        if tracker is None:
            return []
        current_frame_id = tracker.frame_id
        pools = list(tracker.tracked_stracks) + list(tracker.lost_stracks)

        detections: List[DetectionResult] = []
        seen_ids: set = set()
        for strack in pools:
            if not strack.is_activated:
                continue
            track_id = int(strack.track_id)
            if track_id in seen_ids:
                continue
            staleness = current_frame_id - strack.frame_id
            if staleness > max_staleness:
                continue
            seen_ids.add(track_id)
            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in strack.xyxy)
            except Exception:
                continue
            class_id = int(strack.cls)
            detections.append(
                DetectionResult(
                    box=[x1, y1, x2, y2],
                    confidence=float(strack.score),
                    class_id=class_id,
                    class_name=class_label(class_id),
                    track_id=track_id,
                    label_id=class_label_id(class_id),
                    extras={'interpolated': staleness > 0},
                )
            )
        return detections

    # ------------------------------------------------------------------
    def update(self, frame: Any) -> List[DetectionResult]:
        """Back-compat convenience: always a full YOLO+track cycle."""
        return self.detect_and_track(frame)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            'model': self.detector.model_name,
            'tracker': self.tracker_name,
            'confidence': round(self.confidence, 2),
            'iou': round(self.iou, 2),
            'inference_ms': self.last_inference_ms,
            'tracker_ms': self.last_tracker_ms,
            'tracks': self.last_track_count,
            'resolution_resets': self.resolution_reset_count,
        }
