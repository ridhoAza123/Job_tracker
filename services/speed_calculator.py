"""Speed estimation.

Layered on purpose, because a single naive frame-to-frame division is what
produces the 0 / 300 / 900 km/h garbage this module replaces:

1. the bottom-centre of the box is tracked, not the centroid, so growth of
   the box as a vehicle approaches does not register as motion;
2. a constant-velocity Kalman filter smooths position and yields velocity;
3. pixel distance is converted with a perspective-aware pixels-per-meter,
   because a pixel near the horizon covers far more road than one up close;
4. samples are median-filtered, then averaged over a short window;
5. an exponential moving average removes the remaining jitter;
6. physically impossible samples (acceleration or speed out of range) are
   discarded rather than displayed.

When entry and exit lines are configured, the authoritative speed is the
line-to-line average: known distance over measured time. The continuous
estimate is what gets displayed while a vehicle is still in transit.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


class _KalmanFilter2D:
    """Constant-velocity Kalman filter over (x, y) with variable timestep."""

    def __init__(self, point: Point, measurement_noise: float = 9.0, process_noise: float = 55.0):
        self.state = np.array([point[0], point[1], 0.0, 0.0], dtype=np.float64)
        self.covariance = np.diag([25.0, 25.0, 900.0, 900.0]).astype(np.float64)
        self.measurement_noise = float(measurement_noise)
        self.process_noise = float(process_noise)
        self._observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )

    def update(self, point: Point, dt: float) -> Tuple[Point, Point]:
        """Advance by ``dt`` and fuse ``point``. Returns (position, velocity)."""
        dt = float(min(max(dt, 1e-3), 1.0))

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Continuous white-noise acceleration model.
        variance = self.process_noise
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        process = np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        ) * variance

        # Predict.
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

        # Correct.
        measurement = np.array([point[0], point[1]], dtype=np.float64)
        noise = np.eye(2, dtype=np.float64) * self.measurement_noise
        innovation = measurement - self._observation @ self.state
        innovation_cov = self._observation @ self.covariance @ self._observation.T + noise
        try:
            gain = self.covariance @ self._observation.T @ np.linalg.inv(innovation_cov)
        except np.linalg.LinAlgError:
            return (self.state[0], self.state[1]), (self.state[2], self.state[3])
        self.state = self.state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - gain @ self._observation) @ self.covariance

        return (self.state[0], self.state[1]), (self.state[2], self.state[3])


class _TrackState:
    """Per-vehicle speed state."""

    __slots__ = (
        'kalman', 'samples', 'last_time', 'last_point', 'smoothed', 'display_speed',
        'entry_side', 'exit_side', 'entry_time', 'exit_time', 'entry_point', 'exit_point',
        'measured_speed', 'measured', 'updated_at', 'frames',
    )

    def __init__(self, point: Point, now: float):
        self.kalman = _KalmanFilter2D(point)
        self.samples: Deque[float] = deque(maxlen=Config.SPEED_HISTORY_SIZE)
        self.last_time = now
        self.last_point = point
        self.smoothed: Optional[float] = None
        self.display_speed = 0.0
        self.entry_side: Optional[float] = None
        self.exit_side: Optional[float] = None
        self.entry_time: Optional[float] = None
        self.exit_time: Optional[float] = None
        self.entry_point: Optional[Point] = None
        self.exit_point: Optional[Point] = None
        self.measured_speed: Optional[float] = None
        self.measured = False
        self.updated_at = now
        self.frames = 0


class SpeedEstimator:
    """Turns tracked boxes into stable km/h values."""

    def __init__(
        self,
        pixels_per_meter: Optional[float] = None,
        min_speed: Optional[float] = None,
        max_speed: Optional[float] = None,
        speed_limit: Optional[float] = None,
        roi=None,
    ):
        self.logger = logger
        self.pixels_per_meter = float(
            pixels_per_meter if pixels_per_meter is not None else Config.PIXELS_PER_METER
        )
        self.min_speed = float(min_speed if min_speed is not None else Config.MIN_SPEED)
        self.max_speed = float(max_speed if max_speed is not None else Config.MAX_SPEED)
        self.speed_limit = float(
            speed_limit if speed_limit is not None else Config.SPEED_LIMIT
        )
        self.roi = roi
        self._tracks: Dict[int, _TrackState] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def configure(
        self,
        pixels_per_meter: Optional[float] = None,
        speed_limit: Optional[float] = None,
    ) -> None:
        if pixels_per_meter is not None and pixels_per_meter > 0:
            self.pixels_per_meter = float(pixels_per_meter)
        if speed_limit is not None and speed_limit > 0:
            self.speed_limit = float(speed_limit)

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()

    # ------------------------------------------------------------------
    def pixels_per_meter_at(self, y: float, height: int) -> float:
        """Perspective-corrected pixels-per-meter at image row ``y``.

        Near the horizon one metre of road spans far fewer pixels than it does
        in the foreground, so the scale is interpolated between the two.
        """
        if height <= 0:
            return self.pixels_per_meter
        ratio = float(min(max(y / float(height), 0.0), 1.0))
        scale = Config.PERSPECTIVE_TOP_SCALE + (
            Config.PERSPECTIVE_BOTTOM_SCALE - Config.PERSPECTIVE_TOP_SCALE
        ) * ratio
        return max(self.pixels_per_meter * scale, 0.05)

    # ------------------------------------------------------------------
    def update(self, detections: List, frame_shape: Tuple[int, int, int]) -> None:
        """Refresh speed state for every detection in this frame."""
        height, width = frame_shape[0], frame_shape[1]
        now = time.time()

        with self._lock:
            for detection in detections:
                track_id = detection.track_id
                if track_id is None:
                    continue
                point = detection.ground_point

                state = self._tracks.get(track_id)
                if state is None:
                    state = _TrackState(point, now)
                    self._tracks[track_id] = state
                    detection.speed = 0.0
                    detection.status = 'Measuring'
                    continue

                dt = now - state.last_time
                if dt <= 1e-3:
                    detection.speed = round(state.display_speed, 1)
                    detection.status = self.status(state.display_speed)
                    detection.measured = state.measured
                    continue

                filtered_point, velocity = state.kalman.update(point, dt)
                state.last_time = now
                state.last_point = point
                state.updated_at = now
                state.frames += 1

                speed_kmh = self._velocity_to_kmh(velocity, filtered_point[1], height)
                if speed_kmh is not None:
                    state.samples.append(speed_kmh)

                state.display_speed = self._aggregate(state)
                self._update_line_crossing(state, filtered_point, width, height, now)

                reported = (
                    state.measured_speed
                    if state.measured and state.measured_speed is not None
                    else state.display_speed
                )
                detection.speed = round(reported, 1)
                detection.measured = state.measured
                detection.status = self.status(reported)

            self._expire(now)

    # ------------------------------------------------------------------
    def _velocity_to_kmh(
        self, velocity: Point, y: float, height: int
    ) -> Optional[float]:
        """Convert filtered pixel velocity into km/h, rejecting nonsense."""
        pixel_speed = math.hypot(velocity[0], velocity[1])  # px/s
        if not math.isfinite(pixel_speed):
            return None
        meters_per_second = pixel_speed / self.pixels_per_meter_at(y, height)
        speed_kmh = meters_per_second * 3.6
        if not math.isfinite(speed_kmh):
            return None
        # Discard physically impossible samples instead of clamping them into
        # the valid range, which would bias the average upwards.
        if speed_kmh > self.max_speed * 1.25:
            return None
        return max(0.0, speed_kmh)

    @staticmethod
    def _aggregate(state: _TrackState) -> float:
        """Median-filter the window, then exponentially smooth the result."""
        if not state.samples:
            return state.display_speed
        if len(state.samples) < Config.SPEED_MIN_SAMPLES:
            return 0.0  # not enough evidence yet; caller shows "measuring"
        window = sorted(state.samples)
        median = window[len(window) // 2]
        # Average the central samples, dropping the extremes as outliers.
        if len(window) >= 5:
            trimmed = window[1:-1]
            average = sum(trimmed) / len(trimmed)
        else:
            average = sum(window) / len(window)
        candidate = (median + average) / 2.0
        if state.smoothed is None:
            state.smoothed = candidate
        else:
            state.smoothed = 0.7 * state.smoothed + 0.3 * candidate
        return float(state.smoothed)

    # ------------------------------------------------------------------
    def _update_line_crossing(
        self, state: _TrackState, point: Point, width: int, height: int, now: float
    ) -> None:
        """Record entry/exit crossings and derive the authoritative speed."""
        if self.roi is None or not self.roi.has_lines:
            return

        entry_side = self.roi.side_for('entry', point, width, height)
        exit_side = self.roi.side_for('exit', point, width, height)
        if entry_side is None or exit_side is None:
            return

        if state.entry_side is None:
            state.entry_side, state.exit_side = entry_side, exit_side
            return

        # A sign flip means the vehicle passed through that line.
        if state.entry_time is None and _crossed(state.entry_side, entry_side):
            state.entry_time = now
            state.entry_point = point
        if (
            state.entry_time is not None
            and state.exit_time is None
            and _crossed(state.exit_side, exit_side)
        ):
            state.exit_time = now
            state.exit_point = point
            self._finalise_measurement(state, height)

        state.entry_side, state.exit_side = entry_side, exit_side

    def _finalise_measurement(self, state: _TrackState, height: int) -> None:
        """Average speed between the two line crossings."""
        if state.entry_point is None or state.exit_point is None:
            return
        if state.entry_time is None or state.exit_time is None:
            return
        elapsed = state.exit_time - state.entry_time
        if elapsed <= 0.05:  # implausibly fast crossing: ignore
            return

        dx = state.exit_point[0] - state.entry_point[0]
        dy = state.exit_point[1] - state.entry_point[1]
        distance_pixels = math.hypot(dx, dy)
        mid_y = (state.entry_point[1] + state.exit_point[1]) / 2.0
        distance_meters = distance_pixels / self.pixels_per_meter_at(mid_y, height)
        speed_kmh = (distance_meters / elapsed) * 3.6

        if not math.isfinite(speed_kmh) or speed_kmh > self.max_speed * 1.25:
            self.logger.debug('Rejected line measurement of %.1f km/h', speed_kmh)
            return
        state.measured_speed = float(speed_kmh)
        state.measured = True

    # ------------------------------------------------------------------
    def status(self, speed: float) -> str:
        # Below MIN_SPEED we cannot tell "still warming up" from "stopped",
        # and neither is recorded, so both read as Measuring.
        if speed < self.min_speed:
            return 'Measuring'
        if speed > self.speed_limit:
            return 'Overspeed'
        return 'Normal'

    def is_recordable(self, detection) -> bool:
        """True when a detection's speed is trustworthy enough to store.

        With entry/exit lines configured only line-confirmed measurements
        qualify, which is what makes the stored numbers defensible.
        """
        speed = float(getattr(detection, 'speed', 0.0) or 0.0)
        if not (self.min_speed <= speed <= self.max_speed):
            return False
        if self.roi is not None and self.roi.has_lines:
            return bool(getattr(detection, 'measured', False))
        state = self._tracks.get(getattr(detection, 'track_id', None))
        return state is not None and len(state.samples) >= Config.SPEED_MIN_SAMPLES

    def history_for(self, track_id: int) -> List[float]:
        with self._lock:
            state = self._tracks.get(track_id)
            return list(state.samples) if state else []

    def _expire(self, now: float) -> None:
        stale = [
            track_id
            for track_id, state in self._tracks.items()
            if now - state.updated_at > Config.TRACK_EXPIRY_SECONDS
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)

    @property
    def tracked_count(self) -> int:
        with self._lock:
            return len(self._tracks)

    def crossing_stats(self) -> dict:
        """Diagnostics for the entry/exit gate.

        If ``entry_crossings`` climbs while ``measured`` stays at zero, the
        exit line is placed where vehicles never reach it (or they leave the
        ROI first) and should be moved.
        """
        with self._lock:
            states = list(self._tracks.values())
        return {
            'lines_active': bool(self.roi is not None and self.roi.has_lines),
            'entry_crossings': sum(1 for state in states if state.entry_time is not None),
            'exit_crossings': sum(1 for state in states if state.exit_time is not None),
            'measured_tracks': sum(1 for state in states if state.measured),
        }


def _crossed(previous_side: float, current_side: float) -> bool:
    """True when the signed side changed, i.e. the line was passed."""
    if previous_side == 0.0 or current_side == 0.0:
        return False
    return (previous_side > 0.0) != (current_side > 0.0)
