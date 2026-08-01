"""Region of interest and the entry/exit measurement lines.

Coordinates are stored normalised (0..1) so a region drawn on the dashboard
keeps its meaning when the camera resolution changes.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def _clamp01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def _parse_points(raw: Optional[Sequence]) -> List[Point]:
    """Accept [[x,y], …] or [{'x':…,'y':…}, …]; ignore malformed entries."""
    points: List[Point] = []
    for item in raw or []:
        try:
            if isinstance(item, dict):
                x, y = item.get('x'), item.get('y')
            else:
                x, y = item[0], item[1]
            if x is None or y is None:
                continue
            points.append((_clamp01(x), _clamp01(y)))
        except (TypeError, ValueError, IndexError):
            continue
    return points


def side_of_line(point: Point, line: Sequence[Point]) -> float:
    """Signed side of ``point`` relative to the directed line (2-D cross product)."""
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class RegionOfInterest:
    """Thread-safe ROI polygon plus optional entry and exit lines."""

    def __init__(self):
        self._lock = threading.RLock()
        self._polygon: List[Point] = []
        self._entry_line: List[Point] = []
        self._exit_line: List[Point] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def update(
        self,
        polygon: Optional[Sequence] = None,
        entry_line: Optional[Sequence] = None,
        exit_line: Optional[Sequence] = None,
    ) -> dict:
        """Replace whichever elements were supplied. Returns the new state."""
        with self._lock:
            if polygon is not None:
                points = _parse_points(polygon)
                # Fewer than three points cannot bound an area: treat as clear.
                self._polygon = points if len(points) >= 3 else []
            if entry_line is not None:
                points = _parse_points(entry_line)
                self._entry_line = points[:2] if len(points) >= 2 else []
            if exit_line is not None:
                points = _parse_points(exit_line)
                self._exit_line = points[:2] if len(points) >= 2 else []
        return self.to_dict()

    def clear(self) -> dict:
        with self._lock:
            self._polygon = []
            self._entry_line = []
            self._exit_line = []
        return self.to_dict()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                'polygon': [list(point) for point in self._polygon],
                'entry_line': [list(point) for point in self._entry_line],
                'exit_line': [list(point) for point in self._exit_line],
                'has_polygon': len(self._polygon) >= 3,
                'has_lines': len(self._entry_line) == 2 and len(self._exit_line) == 2,
            }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def has_polygon(self) -> bool:
        with self._lock:
            return len(self._polygon) >= 3

    @property
    def has_lines(self) -> bool:
        with self._lock:
            return len(self._entry_line) == 2 and len(self._exit_line) == 2

    def polygon_pixels(self, width: int, height: int) -> Optional[np.ndarray]:
        with self._lock:
            if len(self._polygon) < 3:
                return None
            points = [(x * width, y * height) for x, y in self._polygon]
        return np.array(points, dtype=np.int32)

    def line_pixels(self, name: str, width: int, height: int) -> Optional[List[Tuple[int, int]]]:
        with self._lock:
            line = self._entry_line if name == 'entry' else self._exit_line
            if len(line) != 2:
                return None
            return [(int(x * width), int(y * height)) for x, y in line]

    def contains(self, point: Point, width: int, height: int) -> bool:
        """True when the point is inside the ROI (or no ROI is configured)."""
        polygon = self.polygon_pixels(width, height)
        if polygon is None:
            return True
        try:
            return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0
        except Exception:
            return True

    def side_for(self, name: str, point: Point, width: int, height: int) -> Optional[float]:
        """Signed side of ``point`` for the named line, or None if unset."""
        line = self.line_pixels(name, width, height)
        if line is None:
            return None
        return side_of_line(point, line)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the ROI and the measurement lines onto ``frame``."""
        if frame is None:
            return frame
        height, width = frame.shape[:2]

        polygon = self.polygon_pixels(width, height)
        if polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (60, 180, 75))
            cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
            cv2.polylines(frame, [polygon], True, (80, 220, 120), 2)

        for name, colour, caption in (
            ('entry', (0, 200, 255), 'ENTRY'),
            ('exit', (255, 120, 0), 'EXIT'),
        ):
            line = self.line_pixels(name, width, height)
            if line is None:
                continue
            cv2.line(frame, line[0], line[1], colour, 3)
            label_x = min(max(line[0][0], 5), max(width - 70, 5))
            label_y = min(max(line[0][1] - 8, 16), height - 5)
            cv2.putText(
                frame, caption, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
            )
        return frame
