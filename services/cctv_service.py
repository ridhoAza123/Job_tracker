"""Camera registry and capture orchestration.

Owns the discovered camera list, the currently selected camera and the single
background :class:`~services.streamer.Streamer`. Switching cameras retargets
the existing streamer, so the Flask server is never restarted.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from config import Config
from services.cctv_fetcher import (
    STATUS_OFFLINE,
    STATUS_ONLINE,
    STATUS_UNKNOWN,
    CameraInfo,
    CCTVCrawler,
    classify_stream_type,
    is_direct_stream_url,
)
from services.streamer import Streamer

logger = logging.getLogger(__name__)


class CCTVService:
    """Discovery + selection + capture for the CCTV portal."""

    def __init__(self, source_url: Optional[str] = None):
        self.logger = logger
        self.source_url = (source_url or Config.CCTV_STREAM_URL or '').strip()
        self.crawler = CCTVCrawler(self.source_url, self.logger)
        self.streamer = Streamer(self.logger, on_state_change=self._on_stream_state)

        self._cameras: List[CameraInfo] = []
        self._lock = threading.RLock()
        self._refreshing = False

        self.current_camera: Optional[CameraInfo] = None
        self.last_discovery_error: Optional[str] = None
        self.last_discovery_at: Optional[float] = None
        self.using_fallback = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def refresh_cameras(self, probe: bool = True) -> List[CameraInfo]:
        """Re-crawl the portal. Falls back to CCTV_STREAM_URL on failure."""
        with self._lock:
            if self._refreshing:
                return list(self._cameras)
            self._refreshing = True
        try:
            cameras: List[CameraInfo] = []
            if Config.AUTO_FETCH_CAMERAS and self.source_url:
                try:
                    cameras = self.crawler.discover(probe=probe)
                except Exception as error:
                    self.logger.error('Camera discovery failed: %s', error)
                    self.last_discovery_error = str(error)

            if cameras:
                self.using_fallback = False
                self.last_discovery_error = None
            else:
                cameras = self._fallback_cameras()
                self.using_fallback = True
                self.last_discovery_error = (
                    self.crawler.last_error or self.last_discovery_error
                    or 'No cameras found while crawling'
                )
                self.logger.warning(
                    'Using fallback camera from CCTV_STREAM_URL (%s)',
                    self.last_discovery_error,
                )

            with self._lock:
                self._cameras = cameras
                self.last_discovery_at = time.time()
                # Keep the selected camera's metadata fresh, or pick one.
                if self.current_camera is not None:
                    match = self._find(self.current_camera.url)
                    if match is not None:
                        self.current_camera = match
            return list(cameras)
        finally:
            with self._lock:
                self._refreshing = False

    def refresh_cameras_async(self, probe: bool = True) -> None:
        """Refresh in the background so HTTP requests stay responsive."""
        threading.Thread(
            target=self.refresh_cameras, kwargs={'probe': probe},
            name='camera-refresh', daemon=True,
        ).start()

    def _fallback_cameras(self) -> List[CameraInfo]:
        """The entry point itself, used when crawling yields nothing."""
        if not self.source_url:
            return []
        return [
            CameraInfo(
                url=self.source_url,
                name='CCTV_STREAM_URL (fallback)',
                status=STATUS_UNKNOWN if is_direct_stream_url(self.source_url) else STATUS_OFFLINE,
                reason=(
                    'Fallback to CCTV_STREAM_URL; crawling found no cameras'
                    if is_direct_stream_url(self.source_url)
                    else 'CCTV_STREAM_URL is a portal page, not a playable stream'
                ),
                stream_type=classify_stream_type(self.source_url),
                source='fallback',
            )
        ]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Discover cameras and begin streaming the first usable one."""
        self.refresh_cameras(probe=True)
        camera = self._first_usable()
        if camera is None:
            self.logger.error('No usable camera to start streaming.')
            return
        self.select_camera(camera.url)

    def select_camera(self, camera_url: str) -> bool:
        """Switch the live feed to ``camera_url`` without a restart."""
        camera_url = (camera_url or '').strip()
        if not camera_url:
            return False

        camera = self._find(camera_url)
        if camera is None:
            if not is_direct_stream_url(camera_url):
                self.logger.warning('Refusing to select non-stream URL: %s', camera_url)
                return False
            # Allow selecting a URL that was not part of the crawl result.
            camera = CameraInfo(
                url=camera_url,
                name=CCTVCrawler._name_from_url(camera_url),
                stream_type=classify_stream_type(camera_url),
                source='manual',
            )
            with self._lock:
                self._cameras.append(camera)

        with self._lock:
            self.current_camera = camera
        self.streamer.set_url(camera.url)
        self.logger.info('Selected camera: %s (%s)', camera.name, camera.url)
        return True

    def stop(self) -> None:
        self.streamer.stop()

    def _first_usable(self) -> Optional[CameraInfo]:
        """Prefer a probed-online camera, else anything playable."""
        with self._lock:
            cameras = list(self._cameras)
        for camera in cameras:
            if camera.status == STATUS_ONLINE:
                return camera
        for camera in cameras:
            if camera.status == STATUS_UNKNOWN and is_direct_stream_url(camera.url):
                return camera
        for camera in cameras:
            if is_direct_stream_url(camera.url):
                return camera
        return None

    def _find(self, camera_url: str) -> Optional[CameraInfo]:
        key = (camera_url or '').rstrip('/')
        with self._lock:
            for camera in self._cameras:
                if camera.url.rstrip('/') == key:
                    return camera
        return None

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def read_latest(self, last_frame_id: int = -1):
        return self.streamer.read_latest(last_frame_id)

    def read(self):
        return self.streamer.read()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def cameras(self) -> List[CameraInfo]:
        with self._lock:
            return list(self._cameras)

    @property
    def current_camera_name(self) -> Optional[str]:
        camera = self.current_camera
        return camera.name if camera else None

    @property
    def current_camera_url(self) -> Optional[str]:
        camera = self.current_camera
        return camera.url if camera else None

    @property
    def current_status(self) -> str:
        """Live capture state, which supersedes the crawl-time probe."""
        state = self.streamer.state
        if state == 'ONLINE':
            return STATUS_ONLINE
        camera = self.current_camera
        if camera is not None and camera.status not in (STATUS_ONLINE, STATUS_UNKNOWN):
            return camera.status  # TOKEN REQUIRED / NOT FOUND survive
        return state

    def status_payload(self) -> Dict[str, object]:
        """Everything the dashboard's status panel needs."""
        camera = self.current_camera
        stream_stats = self.streamer.stats()
        live_resolution = self.streamer.resolution
        return {
            'cctv_status': self.current_status,
            'capture_state': stream_stats['state'],
            'camera_name': camera.name if camera else None,
            'camera_id': camera.camera_id if camera else None,
            'stream_url': camera.url if camera else None,
            'stream_type': camera.stream_type if camera else None,
            'reason': (
                stream_stats['last_error']
                or (camera.reason if camera else 'No camera selected')
            ),
            'resolution': (
                live_resolution if live_resolution != '-'
                else (camera.resolution if camera else '-')
            ),
            'source_fps': stream_stats['source_fps'],
            'capture_fps': stream_stats['capture_fps'],
            'reconnect_attempt': stream_stats['reconnect_attempt'],
            'reconnect_count': stream_stats['reconnect_count'],
            'frames_read': stream_stats['frames_read'],
            'frames_dropped': stream_stats['frames_dropped'],
            'buffer_size': stream_stats['buffer_size'],
            'buffer_capacity': stream_stats['buffer_capacity'],
            'uptime': stream_stats['uptime'],
            'camera_count': len(self.cameras),
            'online_count': sum(1 for c in self.cameras if c.status == STATUS_ONLINE),
            'using_fallback': self.using_fallback,
            'discovery_error': self.last_discovery_error,
            'source_url': self.source_url,
        }

    def cameras_payload(self) -> Dict[str, object]:
        cameras = self.cameras
        return {
            'cameras': [camera.to_dict() for camera in cameras],
            'count': len(cameras),
            'online': sum(1 for c in cameras if c.status == STATUS_ONLINE),
            'selected_url': self.current_camera_url,
            'using_fallback': self.using_fallback,
            'discovery_error': self.last_discovery_error,
            'last_refresh': self.last_discovery_at,
        }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_stream_state(self, state: str, reason: str) -> None:
        camera = self.current_camera
        name = camera.name if camera else 'unknown'
        if state == 'ONLINE':
            self.logger.info('Camera Connected: %s', name)
        elif state in ('RECONNECTING', 'FAILED'):
            self.logger.warning('Camera Disconnected: %s (%s)', name, reason)
