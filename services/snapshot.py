"""Snapshot storage.

Images are written only for genuine overspeed events or an explicit manual
capture, and each track may only produce one automatic snapshot. That is what
keeps the folder from filling with thousands of near-identical frames.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from config import BASE_DIR, Config

logger = logging.getLogger(__name__)


class SnapshotService:
    def __init__(self, snapshot_dir: Optional[str] = None):
        self.logger = logger
        self.snapshot_dir = Path(snapshot_dir or Config.SNAPSHOT_DIR)
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self.logger.error('Cannot create snapshot directory: %s', error)
        self._captured_tracks: dict[int, float] = {}
        self._lock = threading.Lock()
        self.saved_count = 0

    # ------------------------------------------------------------------
    def should_capture(self, track_id: Optional[int], is_overspeed: bool) -> bool:
        """One automatic snapshot per track, and only when overspeeding."""
        if not is_overspeed or track_id is None:
            return False
        now = time.time()
        with self._lock:
            last = self._captured_tracks.get(track_id)
            if last is not None and now - last < Config.SNAPSHOT_COOLDOWN_SECONDS:
                return False
            if last is not None:
                return False  # already captured this vehicle
            self._captured_tracks[track_id] = now
        return True

    # ------------------------------------------------------------------
    def save(
        self,
        frame,
        track_id: Optional[int] = None,
        box: Optional[list] = None,
        prefix: str = 'vehicle',
    ) -> Optional[str]:
        """Write a snapshot and return its web-relative path, or None."""
        if frame is None:
            return None
        image = frame
        if box is not None:
            image = self._crop_with_context(frame, box)
        if image is None or getattr(image, 'size', 0) == 0:
            image = frame

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        suffix = f'_{prefix}{track_id}' if track_id is not None else f'_{prefix}'
        path = self.snapshot_dir / f'{stamp}{suffix}.jpg'
        try:
            ok = cv2.imwrite(
                str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
        except Exception as error:
            self.logger.error('Snapshot write failed: %s', error)
            return None
        if not ok:
            self.logger.error('Snapshot could not be encoded: %s', path)
            return None

        self.saved_count += 1
        self.logger.info('Snapshot saved: %s', path.name)
        return self._web_path(path)

    # ------------------------------------------------------------------
    @staticmethod
    def _crop_with_context(frame, box: list, margin: float = 0.25):
        """Crop the vehicle with margin so the context stays visible."""
        try:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = (int(value) for value in box[:4])
            pad_x = int((x2 - x1) * margin)
            pad_y = int((y2 - y1) * margin)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                return None
            return frame[y1:y2, x1:x2]
        except Exception:
            return None

    @staticmethod
    def _web_path(path: Path) -> str:
        """Path relative to the project root, using forward slashes."""
        try:
            return path.resolve().relative_to(BASE_DIR).as_posix()
        except Exception:
            return path.as_posix()

    def forget(self, track_id: int) -> None:
        with self._lock:
            self._captured_tracks.pop(track_id, None)

    def prune(self, max_age: float = 600.0) -> None:
        """Drop bookkeeping for tracks that disappeared long ago."""
        now = time.time()
        with self._lock:
            stale = [
                track_id for track_id, stamp in self._captured_tracks.items()
                if now - stamp > max_age
            ]
            for track_id in stale:
                self._captured_tracks.pop(track_id, None)
