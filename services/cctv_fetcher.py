"""CCTV discovery.

Crawls the source portal and returns *every* camera it can find. The portal
renders its camera list in two different ways (a Leaflet map whose markers are
built inside ``<script>`` template literals, and a plain grid page), so the
crawler parses the real DOM *and* the text of every script with the same
generic extractor, then falls back to raw regex over the whole payload plus
any external JS bundle and AJAX endpoint it can spot.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import Config

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Status vocabulary
# ----------------------------------------------------------------------------
STATUS_ONLINE = 'ONLINE'
STATUS_OFFLINE = 'OFFLINE'
STATUS_TOKEN_REQUIRED = 'TOKEN REQUIRED'
STATUS_NOT_FOUND = 'NOT FOUND'
STATUS_UNKNOWN = 'UNKNOWN'

# ----------------------------------------------------------------------------
# Patterns we hunt for, per the discovery contract.
# ----------------------------------------------------------------------------
STREAM_EXTENSIONS = ('.m3u8', '.mpd', '.flv', '.mp4', '.mjpg', '.mjpeg', '.ts')
STREAM_SCHEMES = ('rtsp://', 'rtmp://', 'rtmps://', 'webrtc://')
STREAM_KEYWORDS = ('webrtc', 'playlist', 'stream', 'camera', 'cctv', 'live', 'hls', 'dash')

# Absolute / rooted URLs that end in a stream extension or use a stream scheme.
STREAM_URL_REGEX = re.compile(
    r'''(?:(?:rtsp|rtmps?|webrtc)://[^\s"'`<>\\)]+)'''
    r'''|(?:https?://[^\s"'`<>\\)]+?(?:\.m3u8|\.mpd|\.flv|\.mp4|\.mjpe?g)(?:\?[^\s"'`<>\\)]*)?)'''
    r'''|(?:/[^\s"'`<>\\)]+?(?:\.m3u8|\.mpd|\.flv|\.mjpe?g)(?:\?[^\s"'`<>\\)]*)?)''',
    re.IGNORECASE,
)

# Attributes that habitually carry a media URL.
URL_ATTRIBUTES = (
    'data-url', 'data-src', 'data-stream', 'data-stream-url', 'data-video',
    'data-video-url', 'data-hls', 'data-source', 'data-file', 'data-href',
    'data-link', 'data-play', 'src', 'href',
)

# Endpoints referenced from JS that may return a camera list.
AJAX_HINT_REGEX = re.compile(
    r'''(?:fetch|axios(?:\.\w+)?|\$\.(?:get|post|ajax)|url\s*:)\s*\(?\s*["'`]([^"'`]+)["'`]''',
    re.IGNORECASE,
)
JS_ASSET_REGEX = re.compile(r'''["'\(]([^"'\(\)\s]+?\.js(?:\?[^"'\)\s]*)?)["'\)]''', re.IGNORECASE)

# Resolution / frame-rate advertised by an HLS master playlist.
HLS_RESOLUTION_REGEX = re.compile(r'RESOLUTION=(\d+x\d+)', re.IGNORECASE)
HLS_FRAMERATE_REGEX = re.compile(r'FRAME-RATE=([\d.]+)', re.IGNORECASE)

TOKEN_HINT_REGEX = re.compile(
    r'\b(token|unauthorized|forbidden|auth|signature|expired|denied)\b', re.IGNORECASE
)

_NAME_TAGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'strong', 'figcaption', 'legend')
_NOISE_NAME = re.compile(r'^(thumbnail|play|image|video|cctv|untitled|)$', re.IGNORECASE)


@dataclass
class CameraInfo:
    """A discovered camera and everything the dashboard needs to show it."""

    url: str
    name: str
    camera_id: Optional[str] = None
    status: str = STATUS_UNKNOWN
    reason: str = 'Not probed yet'
    resolution: Optional[str] = None
    frame_rate: Optional[float] = None
    stream_type: str = 'UNKNOWN'
    source: str = 'crawl'
    variant_url: Optional[str] = None
    extras: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'url': self.url,
            'name': self.name,
            'label': self.name,
            'camera_id': self.camera_id,
            'status': self.status,
            'reason': self.reason,
            'resolution': self.resolution or '-',
            'frame_rate': self.frame_rate,
            'stream_type': self.stream_type,
            'source': self.source,
            'online': self.status == STATUS_ONLINE,
        }


def classify_stream_type(url: str, content_type: str = '') -> str:
    """Best-effort stream family from the URL and/or response content type."""
    lowered = (url or '').lower()
    content_type = (content_type or '').lower()
    if lowered.startswith('rtsp://'):
        return 'RTSP'
    if lowered.startswith(('rtmp://', 'rtmps://')):
        return 'RTMP'
    if lowered.startswith('webrtc://') or 'webrtc' in lowered:
        return 'WEBRTC'
    if '.m3u8' in lowered or 'mpegurl' in content_type:
        return 'HLS'
    if '.mpd' in lowered or 'dash+xml' in content_type:
        return 'DASH'
    if '.flv' in lowered or 'x-flv' in content_type:
        return 'FLV'
    if '.mjpg' in lowered or '.mjpeg' in lowered or 'multipart/x-mixed-replace' in content_type:
        return 'MJPEG'
    if '.mp4' in lowered or 'video/mp4' in content_type:
        return 'MP4'
    if lowered.startswith(('http://', 'https://')):
        return 'HTTP'
    return 'UNKNOWN'


def is_direct_stream_url(url: str) -> bool:
    """True when the URL can be handed straight to VideoCapture."""
    if not url:
        return False
    lowered = url.strip().lower()
    if lowered.startswith(STREAM_SCHEMES):
        return True
    if not lowered.startswith(('http://', 'https://')):
        return False
    path = urlparse(lowered).path
    return any(ext in path for ext in STREAM_EXTENSIONS)


class CCTVCrawler:
    """Discovers cameras from a portal entry point."""

    def __init__(self, source_url: str, logger_: Optional[logging.Logger] = None):
        self.source_url = (source_url or '').strip()
        self.logger = logger_ or logger
        self.session = requests.Session()
        origin = self._origin(self.source_url)
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
        })
        if origin:
            # Cookies, referer and origin are carried on every request.
            self.session.headers.update({'Referer': origin + '/', 'Origin': origin})
        # Sized for the parallel probe fan-out so connections are reused
        # instead of being discarded.
        pool_size = max(10, Config.CCTV_MAX_PROBE_WORKERS + 4)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.last_error: Optional[str] = None
        self._visited: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover(self, probe: bool = False) -> List[CameraInfo]:
        """Return every camera found at the entry point.

        ``probe`` additionally contacts each stream to resolve status and
        resolution. Discovery never raises; on total failure it returns [].
        """
        self.last_error = None
        self._visited.clear()

        if not self.source_url:
            self.last_error = 'CCTV_STREAM_URL is empty'
            return []

        # A source that is already a playable stream needs no crawling.
        if is_direct_stream_url(self.source_url):
            camera = CameraInfo(
                url=self.source_url,
                name=self._name_from_url(self.source_url),
                stream_type=classify_stream_type(self.source_url),
                source='direct',
            )
            return self._probe_all([camera]) if probe else [camera]

        found: Dict[str, CameraInfo] = {}
        for page in self._candidate_pages():
            self._crawl_page(page, found)

        cameras = [camera for camera in found.values() if is_direct_stream_url(camera.url)]
        cameras.sort(key=lambda camera: (camera.name or '').lower())
        if not cameras:
            self.logger.warning('No cameras discovered at %s', self.source_url)
        else:
            self.logger.info('Discovered %d camera(s) from %s', len(cameras), self.source_url)
        return self._probe_all(cameras) if probe else cameras

    def _candidate_pages(self) -> List[str]:
        """Entry point first, then the site root.

        The portal publishes the same cameras on a grid page and on a map
        page, but only the map page carries untruncated names, so merging
        both yields the best metadata.
        """
        pages = [self.source_url]
        origin = self._origin(self.source_url)
        if origin:
            root = origin + '/'
            if root.rstrip('/') != self.source_url.rstrip('/'):
                pages.append(root)
        return pages

    def probe(self, cameras: Sequence[CameraInfo]) -> List[CameraInfo]:
        """Resolve status/resolution for already-discovered cameras."""
        return self._probe_all(list(cameras))

    # ------------------------------------------------------------------
    # Crawling
    # ------------------------------------------------------------------
    def _crawl_page(self, url: str, found: Dict[str, CameraInfo]) -> None:
        """Harvest every camera reachable from one portal page into ``found``."""
        page = self._fetch(url)
        if page is None:
            return
        html, final_url, _ = page

        # 1) The rendered DOM.
        soup = self._soup(html)
        self._collect_from_soup(soup, final_url, 'html', found)

        # 2) Every inline script, re-parsed as an HTML fragment. The portal
        #    builds map popups from template literals, so the same generic
        #    extractor finds them once the script text is treated as markup.
        script_bodies: List[str] = []
        for script in soup.find_all('script'):
            body = script.string or script.get_text() or ''
            if body.strip():
                script_bodies.append(body)
        for body in script_bodies:
            if any(token in body.lower() for token in STREAM_KEYWORDS) or 'data-url' in body:
                self._collect_from_soup(self._soup(body), final_url, 'inline-js', found)

        # 3) Raw regex over the whole document (catches anything the parsers
        #    cannot reach, e.g. escaped or concatenated URLs).
        self._collect_from_text(html, final_url, 'regex', found)

        # 4) External JS bundles.
        for asset in self._javascript_assets(html, final_url):
            body = self._fetch_text(asset)
            if body:
                self._collect_from_text(body, asset, 'js-asset', found)
                self._collect_from_soup(self._soup(body), final_url, 'js-asset-dom', found)

        # 5) AJAX endpoints referenced by the page/scripts.
        for endpoint in self._ajax_endpoints(html, script_bodies, final_url):
            body = self._fetch_text(endpoint)
            if body:
                self._collect_from_text(body, endpoint, 'ajax', found)
                if '<' in body:
                    self._collect_from_soup(self._soup(body), endpoint, 'ajax-dom', found)

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _soup(markup: str) -> BeautifulSoup:
        return BeautifulSoup(markup, 'html.parser')

    def _collect_from_soup(
        self, soup: BeautifulSoup, base_url: str, source: str, found: Dict[str, CameraInfo]
    ) -> None:
        """Pull stream URLs out of a parsed tree, naming each from its DOM."""
        for element in soup.find_all(True):
            for attribute in URL_ATTRIBUTES:
                raw = element.get(attribute)
                if not raw or not isinstance(raw, str):
                    continue
                url = self._normalize(raw, base_url)
                if not url or not is_direct_stream_url(url):
                    continue
                self._register(
                    found,
                    url=url,
                    name=self._name_for_element(element, url),
                    camera_id=element.get('data-id') or element.get('id'),
                    source=source,
                )

    def _collect_from_text(
        self, text: str, base_url: str, source: str, found: Dict[str, CameraInfo]
    ) -> None:
        """Regex sweep; names are derived from the URL slug."""
        for match in STREAM_URL_REGEX.finditer(text or ''):
            url = self._normalize(match.group(0), base_url)
            if not url or not is_direct_stream_url(url):
                continue
            self._register(found, url=url, name=self._name_from_url(url), source=source)

    def _register(
        self,
        found: Dict[str, CameraInfo],
        url: str,
        name: str,
        source: str,
        camera_id: Optional[str] = None,
    ) -> None:
        """Insert a camera, letting a better name/id win on duplicates."""
        key = url.rstrip('/')
        existing = found.get(key)
        if existing is None:
            found[key] = CameraInfo(
                url=url,
                name=name or self._name_from_url(url),
                camera_id=str(camera_id) if camera_id else None,
                stream_type=classify_stream_type(url),
                source=source,
            )
            return
        # Keep whichever name is more informative.
        if name and not _NOISE_NAME.match(name):
            if self._name_score(name) > self._name_score(existing.name):
                existing.name = name
                existing.source = source
        if camera_id and not existing.camera_id:
            existing.camera_id = str(camera_id)

    @staticmethod
    def _name_score(name: str) -> int:
        """Rank candidate names: real words beat slugs, full beats truncated."""
        if not name:
            return -1
        score = len(name)
        if name.endswith(('...', '…')):
            score -= 50  # server-side truncated label
        return score

    def _name_for_element(self, element, url: str) -> str:
        """Human name for a stream element, taken from the nearest label.

        Resolution order matters: the *closest preceding* label wins. When a
        whole script body is parsed as one fragment every camera ends up in a
        single flat tree, so searching a parent subtree top-down would give
        every camera the first camera's name.
        """
        # Explicit attributes first.
        for attribute in ('data-name', 'data-title', 'data-label', 'title', 'alt', 'aria-label'):
            value = element.get(attribute)
            if isinstance(value, str) and value.strip() and not _NOISE_NAME.match(value.strip()):
                return self._clean_name(value)

        # Nearest preceding label, walking outwards level by level.
        node = element
        for _ in range(5):
            if node is None:
                break
            for sibling in node.previous_siblings:
                if getattr(sibling, 'name', None) is None:
                    continue  # NavigableString
                text = self._label_within(sibling)
                if text:
                    return text
            node = node.parent

        # Last resort: any label inside the immediate parent block.
        parent = element.parent
        if parent is not None:
            text = self._label_within(parent)
            if text:
                return text

        return self._name_from_url(url)

    def _label_within(self, node) -> Optional[str]:
        """Text of the last label at/inside ``node``, or None."""
        candidates = []
        if getattr(node, 'name', None) in _NAME_TAGS:
            candidates.append(node)
        try:
            candidates.extend(node.find_all(_NAME_TAGS, recursive=True))
        except Exception:
            pass
        # Later candidates sit closer to the stream element.
        for tag in reversed(candidates):
            text = self._clean_name(tag.get_text(' ', strip=True))
            if text and not _NOISE_NAME.match(text):
                return text
        return None

    @staticmethod
    def _clean_name(value: str) -> str:
        text = re.sub(r'\s+', ' ', (value or '')).strip()
        return text[:120]

    @staticmethod
    def _name_from_url(url: str) -> str:
        """Derive a readable name from a stream URL slug."""
        try:
            path = urlparse(url).path
        except Exception:
            path = url
        parts = [part for part in Path(path).parts if part not in ('/', '\\')]
        slug = ''
        for part in reversed(parts):
            stem = Path(part).stem
            # Skip generic playlist filenames like index / master / playlist.
            if stem.lower() in {'index', 'master', 'playlist', 'live', 'stream', 'chunklist'}:
                continue
            slug = stem
            break
        if not slug:
            slug = urlparse(url).netloc or url
        return re.sub(r'[_\-]+', ' ', slug).strip().upper()[:120] or 'CAMERA'

    def _javascript_assets(self, html: str, base_url: str) -> List[str]:
        """Same-origin JS files worth downloading."""
        assets: List[str] = []
        base_host = self._host(base_url)
        for match in JS_ASSET_REGEX.finditer(html or ''):
            url = self._normalize(match.group(1), base_url)
            if not url or url in assets:
                continue
            if self._host(url) != base_host:
                continue  # skip CDNs / third-party libraries
            assets.append(url)
        return assets[:8]

    def _ajax_endpoints(
        self, html: str, script_bodies: Iterable[str], base_url: str
    ) -> List[str]:
        """Candidate JSON endpoints that might return a camera list."""
        endpoints: List[str] = []
        base_host = self._host(base_url)
        haystacks = [html or ''] + list(script_bodies)
        for haystack in haystacks:
            for match in AJAX_HINT_REGEX.finditer(haystack):
                raw = match.group(1)
                if raw.startswith(('data:', 'blob:', 'javascript:', '#')):
                    continue
                if raw.lower().endswith(('.css', '.png', '.jpg', '.jpeg', '.svg', '.woff', '.woff2', '.ico')):
                    continue
                url = self._normalize(raw, base_url)
                if not url or url in endpoints or self._host(url) != base_host:
                    continue
                if is_direct_stream_url(url):
                    continue  # already handled as a stream
                if not any(token in url.lower() for token in STREAM_KEYWORDS + ('api', 'json', 'list')):
                    continue
                endpoints.append(url)
        return endpoints[:8]

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------
    def _probe_all(self, cameras: List[CameraInfo]) -> List[CameraInfo]:
        if not cameras:
            return cameras
        workers = max(1, min(Config.CCTV_MAX_PROBE_WORKERS, len(cameras)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cctv-probe') as pool:
            futures = {pool.submit(self.probe_one, camera): camera for camera in cameras}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:  # never let probing break discovery
                    camera = futures[future]
                    camera.status = STATUS_OFFLINE
                    camera.reason = f'Probe error: {error}'
        online = sum(1 for camera in cameras if camera.status == STATUS_ONLINE)
        self.logger.info('Probed %d camera(s): %d online', len(cameras), online)
        return cameras

    def probe_one(self, camera: CameraInfo) -> CameraInfo:
        """Resolve status, reason, resolution and stream type for one camera."""
        url = camera.url
        scheme = urlparse(url).scheme.lower()

        # Non-HTTP transports cannot be probed over requests; report unknown
        # rather than lying about availability.
        if scheme in {'rtsp', 'rtmp', 'rtmps', 'webrtc'}:
            camera.status = STATUS_UNKNOWN
            camera.reason = f'{scheme.upper()} cannot be verified over HTTP; will be tried on connect'
            return camera

        try:
            response = self.session.get(
                url, timeout=Config.CCTV_PROBE_TIMEOUT, allow_redirects=True, stream=True
            )
        except requests.Timeout:
            camera.status = STATUS_OFFLINE
            camera.reason = f'Timeout after {Config.CCTV_PROBE_TIMEOUT}s'
            return camera
        except requests.RequestException as error:
            camera.status = STATUS_OFFLINE
            camera.reason = f'Connection failed: {type(error).__name__}'
            return camera

        status_code = response.status_code
        content_type = response.headers.get('Content-Type', '')
        try:
            body = response.text[:8192] if status_code < 400 else response.text[:512]
        except Exception:
            body = ''
        finally:
            response.close()

        camera.stream_type = classify_stream_type(url, content_type)

        if status_code in (401, 403):
            camera.status = STATUS_TOKEN_REQUIRED
            camera.reason = f'HTTP {status_code}: authentication or token required'
            return camera
        if status_code == 404:
            camera.status = STATUS_NOT_FOUND
            camera.reason = 'HTTP 404: stream path does not exist'
            return camera
        if status_code >= 500:
            camera.status = STATUS_OFFLINE
            camera.reason = f'HTTP {status_code}: stream server error'
            return camera
        if status_code >= 400:
            camera.status = STATUS_OFFLINE
            camera.reason = f'HTTP {status_code}'
            return camera

        # 2xx: make sure we actually got media, not a login page.
        if 'text/html' in content_type.lower():
            if TOKEN_HINT_REGEX.search(body):
                camera.status = STATUS_TOKEN_REQUIRED
                camera.reason = 'Received an HTML auth page instead of media'
            else:
                camera.status = STATUS_OFFLINE
                camera.reason = 'Received HTML instead of a media stream'
            return camera

        if camera.stream_type == 'HLS' or '#EXTM3U' in body:
            if '#EXTM3U' not in body:
                camera.status = STATUS_OFFLINE
                camera.reason = 'Playlist did not contain #EXTM3U'
                return camera
            resolution = HLS_RESOLUTION_REGEX.search(body)
            frame_rate = HLS_FRAMERATE_REGEX.search(body)
            if resolution:
                camera.resolution = resolution.group(1)
            if frame_rate:
                try:
                    camera.frame_rate = round(float(frame_rate.group(1)), 2)
                except ValueError:
                    pass
            variant = self._first_variant(body, url)
            if variant:
                camera.variant_url = variant
            if not camera.resolution and variant:
                camera.resolution = self._resolution_from_variant(variant)
            has_media = bool(resolution) or '#EXT-X-STREAM-INF' in body or '#EXTINF' in body
            camera.status = STATUS_ONLINE if has_media else STATUS_OFFLINE
            camera.reason = (
                'Playlist reachable' if has_media else 'Playlist contained no media segments'
            )
            return camera

        camera.status = STATUS_ONLINE
        camera.reason = f'HTTP {status_code} ({content_type or "unknown content type"})'
        return camera

    def _first_variant(self, playlist: str, base_url: str) -> Optional[str]:
        """Resolve the first variant playlist referenced by a master playlist."""
        for line in playlist.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            return self._normalize(line, base_url)
        return None

    def _resolution_from_variant(self, variant_url: str) -> Optional[str]:
        """Some servers only advertise resolution inside the variant playlist."""
        body = self._fetch_text(variant_url, limit=4096)
        if not body:
            return None
        match = HLS_RESOLUTION_REGEX.search(body)
        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _fetch(self, url: str) -> Optional[Tuple[str, str, str]]:
        """GET a page following redirects/cookies. Returns (text, final_url, ctype)."""
        if url in self._visited:
            return None
        self._visited.add(url)
        try:
            response = self.session.get(
                url, timeout=Config.CCTV_CRAWL_TIMEOUT, allow_redirects=True
            )
            response.raise_for_status()
            if response.history:
                self.logger.debug(
                    'Followed %d redirect(s) to %s', len(response.history), response.url
                )
            return response.text, str(response.url), response.headers.get('Content-Type', '')
        except requests.RequestException as error:
            self.last_error = f'{type(error).__name__}: {error}'
            self.logger.error('Failed to fetch %s: %s', url, error)
            return None

    def _fetch_text(self, url: str, limit: int = 2_000_000) -> Optional[str]:
        """GET a sub-resource, tolerating any failure."""
        if url in self._visited:
            return None
        self._visited.add(url)
        try:
            response = self.session.get(url, timeout=Config.CCTV_CRAWL_TIMEOUT)
            if response.status_code >= 400:
                return None
            return response.text[:limit]
        except requests.RequestException:
            return None

    # ------------------------------------------------------------------
    # URL utilities
    # ------------------------------------------------------------------
    def _normalize(self, url: str, base_url: str) -> Optional[str]:
        """Absolutise and clean a URL harvested from markup or script text."""
        if not url or not isinstance(url, str):
            return None
        cleaned = url.strip().strip('\'"`').replace('\\/', '/').strip()
        if not cleaned or cleaned.startswith(('#', 'data:', 'blob:', 'javascript:', 'mailto:')):
            return None
        # Trim trailing punctuation left over from surrounding source code.
        cleaned = cleaned.rstrip('\\,;)]}\'"`')
        if cleaned.startswith('//'):
            scheme = urlparse(base_url).scheme or 'https'
            return f'{scheme}:{cleaned}'
        if cleaned.startswith(STREAM_SCHEMES) or cleaned.startswith(('http://', 'https://')):
            return cleaned
        try:
            return urljoin(base_url or self.source_url, cleaned)
        except Exception:
            return None

    @staticmethod
    def _host(url: str) -> str:
        try:
            return (urlparse(url).netloc or '').lower()
        except Exception:
            return ''

    @staticmethod
    def _origin(url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                return f'{parsed.scheme}://{parsed.netloc}'
        except Exception:
            return None
        return None
