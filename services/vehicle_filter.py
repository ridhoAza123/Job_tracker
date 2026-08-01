"""Post-detection vehicle validity filtering.

Sits between the tracker and everything downstream (speed estimation,
counting, persistence, rendering). The tracker's job is to find and follow
boxes; this module's job is to decide which of those boxes are trustworthy
enough to actually count as a vehicle. A box can clear the YOLO confidence
threshold and still be a false positive — a shadow, a lamp post, a sliver of
a vehicle exiting the frame — so this applies size, shape, frame-edge and
temporal-consistency checks on top, plus a second overlap-suppression pass
for any duplicate boxes the tracker left behind.

Runs entirely on plain ``DetectionResult`` values already produced by
whatever backend the tracker used (BoTSORT/ByteTrack via ``model.track()``,
or the plain-detect fallback) — it neither knows nor cares which one that
was, so it drops in without touching tracker.py, speed_calculator.py or the
database layer at all.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from config import Config

logger = logging.getLogger(__name__)

REJECT_SMALL_BBOX = 'small_bbox'
REJECT_LOW_CONFIDENCE = 'low_confidence'
REJECT_BAD_RATIO = 'bad_ratio'
REJECT_EDGE_OBJECT = 'edge_object'
REJECT_SINGLE_FRAME = 'single_frame'
REJECT_DUPLICATE = 'duplicate_overlap'

_LOG_INTERVAL_SECONDS = 3.0


def _iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class VehicleFilter:
    """Rejects implausible boxes and confirms the rest across frames."""

    def __init__(
        self,
        min_width: int = Config.MIN_BBOX_WIDTH,
        min_height: int = Config.MIN_BBOX_HEIGHT,
        min_area: int = Config.MIN_BBOX_AREA,
        min_aspect_ratio: float = Config.MIN_ASPECT_RATIO,
        max_aspect_ratio: float = Config.MAX_ASPECT_RATIO,
        min_confirm_frames: int = Config.MIN_CONFIRM_FRAMES,
        edge_margin_left: float = Config.EDGE_MARGIN_LEFT,
        edge_margin_right: float = Config.EDGE_MARGIN_RIGHT,
        edge_margin_top: float = Config.EDGE_MARGIN_TOP,
        nms_iou: float = Config.DETECTION_IOU,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_confirm_frames = max(1, min_confirm_frames)
        self.edge_margin_left = edge_margin_left
        self.edge_margin_right = edge_margin_right
        self.edge_margin_top = edge_margin_top
        self.nms_iou = nms_iou

        # Cumulative sighting count per track_id, NOT a strictly-consecutive
        # streak — see _confirm_temporal for why that distinction matters.
        self._hit_counts: Dict[int, int] = {}
        self._last_seen: Dict[int, float] = {}
        self._lock = threading.Lock()

        # Cumulative since the current camera was selected — surfaced via
        # stats() for the dashboard/API.
        self._total_raw = 0
        self._total_valid = 0
        self._total_rejected: Dict[str, int] = defaultdict(int)

        # Resets every _LOG_INTERVAL_SECONDS — purely to keep the periodic
        # log line non-spammy (one summary line, not one per frame).
        self._window_raw = 0
        self._window_valid = 0
        self._window_rejected: Dict[str, int] = defaultdict(int)
        self._log_window_start = time.time()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-camera state (call on camera switch — track ids restart)."""
        with self._lock:
            self._hit_counts.clear()
            self._last_seen.clear()
            self._total_raw = 0
            self._total_valid = 0
            self._total_rejected = defaultdict(int)
            self._window_raw = 0
            self._window_valid = 0
            self._window_rejected = defaultdict(int)
            self._log_window_start = time.time()

    # ------------------------------------------------------------------
    def filter(
        self,
        detections: List,
        frame_shape: Tuple[int, int, int],
        confidence_threshold: Optional[float] = None,
    ) -> List:
        """Return only the detections confirmed as real vehicles."""
        height, width = frame_shape[0], frame_shape[1]
        threshold = (
            confidence_threshold if confidence_threshold is not None
            else Config.DETECTION_CONFIDENCE
        )
        raw_count = len(detections)
        reasons: Dict[str, int] = defaultdict(int)

        quality_passed = []
        for detection in detections:
            reason = self._quality_reject_reason(detection, width, height, threshold)
            if reason is None:
                quality_passed.append(detection)
            else:
                reasons[reason] += 1

        deduped, duplicate_count = self._suppress_duplicates(quality_passed)
        if duplicate_count:
            reasons[REJECT_DUPLICATE] += duplicate_count

        confirmed = self._confirm_temporal(deduped, reasons, time.time())

        with self._lock:
            self._total_raw += raw_count
            self._total_valid += len(confirmed)
            self._window_raw += raw_count
            self._window_valid += len(confirmed)
            for key, value in reasons.items():
                self._total_rejected[key] += value
                self._window_rejected[key] += value
        self._maybe_log()

        return confirmed

    # ------------------------------------------------------------------
    # Per-box quality checks
    # ------------------------------------------------------------------
    def _quality_reject_reason(
        self, detection, width: int, height: int, confidence_threshold: float
    ) -> Optional[str]:
        x1, y1, x2, y2 = detection.box
        box_width = x2 - x1
        box_height = y2 - y1
        area = box_width * box_height

        if box_width < self.min_width or box_height < self.min_height or area < self.min_area:
            return REJECT_SMALL_BBOX

        ratio = box_width / box_height if box_height > 0 else float('inf')
        if ratio < self.min_aspect_ratio or ratio > self.max_aspect_ratio:
            return REJECT_BAD_RATIO

        left_edge = width * self.edge_margin_left
        right_edge = width * (1.0 - self.edge_margin_right)
        top_edge = height * self.edge_margin_top
        if x1 <= left_edge or x2 >= right_edge or y1 <= top_edge:
            return REJECT_EDGE_OBJECT

        # Redundant with the live inference threshold in the normal case
        # (YOLO already only returns boxes at/above it) — kept as a defensive
        # re-check in case the threshold changes between inference and here.
        if detection.confidence < confidence_threshold:
            return REJECT_LOW_CONFIDENCE

        return None

    # ------------------------------------------------------------------
    # Second NMS pass: drop any box that overlaps a higher-confidence one
    # ------------------------------------------------------------------
    def _suppress_duplicates(self, detections: List) -> Tuple[List, int]:
        if len(detections) <= 1:
            return detections, 0
        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List = []
        kept_boxes: List[List[int]] = []
        duplicate_count = 0
        for detection in ordered:
            if any(_iou(detection.box, box) >= self.nms_iou for box in kept_boxes):
                duplicate_count += 1
                continue
            kept.append(detection)
            kept_boxes.append(detection.box)
        return kept, duplicate_count

    # ------------------------------------------------------------------
    # Temporal confirmation: must be seen enough times within a grace window
    # ------------------------------------------------------------------
    def _confirm_temporal(self, detections: List, reasons: Dict[str, int], now: float) -> List:
        """A track must accumulate ``min_confirm_frames`` sightings before
        counting as a real vehicle — anti-flicker, same as before.

        REGRESSION FIX: this used to require ``min_confirm_frames`` strictly
        *consecutive* calls to this method, resetting to zero on any single
        cycle where the track didn't appear. That silently assumed the AI
        loop ticks fast relative to how long a vehicle stays in frame. When
        a processing-cost regression elsewhere slowed real cycles to
        multiple seconds apart (against a typical few-seconds-total dwell
        time in a CCTV frame), most vehicles got exactly one qualifying
        cycle before leaving — never enough to satisfy "consecutive" no
        matter how many raw detections YOLO produced. Confirmed live:
        REJECT_SINGLE_FRAME was 40-55% of all rejections during that
        regression, despite raw detection counts being *higher* than
        before it.
        Now: sightings accumulate cumulatively (not reset by one missed
        cycle) and only expire after ``Config.VEHICLE_FILTER_CONFIRM_
        EXPIRY_SECONDS`` of real wall-clock time with no sighting at all —
        robust to cadence being fast, briefly slow, or irregular, instead of
        silently depending on a specific tick rate holding forever.
        """
        confirmed = []
        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                # No stable identity to measure continuity against (only
                # happens if the tracker itself fell back to plain,
                # untracked detection). Pass through rather than silently
                # discarding every vehicle while already in a degraded mode.
                confirmed.append(detection)
                continue
            hits = self._hit_counts.get(track_id, 0) + 1
            self._hit_counts[track_id] = hits
            self._last_seen[track_id] = now
            if hits >= self.min_confirm_frames:
                confirmed.append(detection)
            else:
                reasons[REJECT_SINGLE_FRAME] += 1

        stale_ids = [
            track_id for track_id, last in self._last_seen.items()
            if now - last > Config.VEHICLE_FILTER_CONFIRM_EXPIRY_SECONDS
        ]
        for track_id in stale_ids:
            self._hit_counts.pop(track_id, None)
            self._last_seen.pop(track_id, None)

        return confirmed

    # ------------------------------------------------------------------
    # Logging + stats
    # ------------------------------------------------------------------
    def _maybe_log(self) -> None:
        now = time.time()
        with self._lock:
            if now - self._log_window_start < _LOG_INTERVAL_SECONDS:
                return
            raw, valid, rejected = self._window_raw, self._window_valid, dict(self._window_rejected)
            self._window_raw = 0
            self._window_valid = 0
            self._window_rejected = defaultdict(int)
            self._log_window_start = now
        if raw == 0:
            return
        total_rejected = sum(rejected.values())
        reason_text = ', '.join(f'{k}={v}' for k, v in sorted(rejected.items())) or 'none'
        logger.info(
            'Raw detections: %d | Valid detections: %d | Rejected detections: %d (%s)',
            raw, valid, total_rejected, reason_text,
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                'raw': self._total_raw,
                'valid': self._total_valid,
                'rejected_total': sum(self._total_rejected.values()),
                'rejected': dict(self._total_rejected),
            }
