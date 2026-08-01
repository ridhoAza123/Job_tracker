"""YOLO model loading and the detection value object.

The weights file is chosen automatically: the path from ``YOLO_WEIGHTS_PATH``
is tried first, then the built-in priority list, so no code has to change when
a different checkpoint is dropped into ``weights/``. If a file exists but
fails to load, the next candidate is tried.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from config import BASE_DIR, Config

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """One tracked detection in a single frame."""

    box: List[int]                 # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str               # canonical English label (car, motorcycle, …)
    track_id: Optional[int] = None
    label_id: str = ''            # Indonesian label used by the UI/database
    speed: float = 0.0
    status: str = 'Normal'
    in_roi: bool = True
    measured: bool = False        # speed confirmed by entry+exit crossing
    extras: dict = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the vehicle meets the road.

        Far more stable than the centroid for speed work, because it does not
        drift when the box grows as the vehicle approaches the camera.
        """
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2.0, float(y2)


def resolve_weights(preferred: Optional[str] = None) -> List[Path]:
    """Candidate weight files, best first, de-duplicated and existing only."""
    candidates: List[Path] = []

    def add(raw: Optional[str]) -> None:
        if not raw:
            return
        path = Path(raw)
        if not path.is_absolute():
            path = BASE_DIR / path
        if path.exists() and path.is_file() and path not in candidates:
            candidates.append(path)

    add(preferred or Config.YOLO_WEIGHTS_PATH)
    for entry in Config.MODEL_PRIORITY:
        add(entry)
    return candidates


class Detector:
    """Loads a YOLO model and runs plain (untracked) inference."""

    def __init__(self, weights_path: Optional[str] = None):
        self.logger = logger
        self.model: Any = None
        self.weights_path: Optional[str] = None
        self.model_name: str = 'none'
        self.class_names: dict = {}
        self.load(weights_path)

    # ------------------------------------------------------------------
    def load(self, weights_path: Optional[str] = None) -> bool:
        """Load the first weights file that works. Returns success."""
        candidates = resolve_weights(weights_path)
        if not candidates:
            self.logger.error(
                'No YOLO weights found. Looked for %s and %s',
                weights_path or Config.YOLO_WEIGHTS_PATH,
                ', '.join(Config.MODEL_PRIORITY),
            )
            return False

        from ultralytics import YOLO  # imported lazily: torch load is slow

        for candidate in candidates:
            try:
                self.logger.info('Loading YOLO weights: %s', candidate.name)
                model = YOLO(str(candidate))
                self.model = model
                self.weights_path = str(candidate)
                self.model_name = candidate.name
                self.class_names = dict(getattr(model, 'names', {}) or {})
                self.logger.info('Model ready: %s', candidate.name)
                return True
            except Exception as error:
                self.logger.error(
                    'Failed to load %s (%s). Trying the next candidate.',
                    candidate.name, error,
                )
        self.logger.error('Every candidate model failed to load.')
        return False

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    def detect(
        self,
        frame: Any,
        confidence: float = Config.DETECTION_CONFIDENCE,
        iou: float = Config.DETECTION_IOU,
        classes: Sequence[int] = Config.VEHICLE_CLASS_IDS,
    ) -> List[DetectionResult]:
        """Detect vehicles in one frame. Returns [] on any failure."""
        if not self.is_ready or frame is None:
            return []
        try:
            # Plain forward pass — never .track(); tracking is a separate
            # stage handled by services/tracker.py further down the pipeline.
            results = self.model(
                frame,
                imgsz=Config.INFERENCE_IMAGE_SIZE,
                conf=float(confidence),
                iou=float(iou),
                classes=list(classes),
                agnostic_nms=Config.DETECTION_AGNOSTIC_NMS,
                max_det=Config.DETECTION_MAX_DET,
                verbose=False,
            )
        except Exception as error:
            self.logger.error('Inference failed: %s', error)
            return []
        return self.parse_results(results)

    # ------------------------------------------------------------------
    @classmethod
    def parse_results(cls, results: Any) -> List[DetectionResult]:
        """Convert ultralytics output into DetectionResult objects."""
        detections: List[DetectionResult] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return detections

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
        except Exception:
            return detections

        track_ids = cls._extract_track_ids(boxes, len(xyxy))
        for box, confidence, class_id, track_id in zip(
            xyxy, confidences, class_ids, track_ids
        ):
            class_id = int(class_id)
            x1, y1, x2, y2 = (int(value) for value in box[:4])
            detections.append(
                DetectionResult(
                    box=[x1, y1, x2, y2],
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=class_label(class_id),
                    track_id=track_id,
                    label_id=class_label_id(class_id),
                )
            )
        return detections

    @staticmethod
    def _extract_track_ids(boxes: Any, count: int) -> List[Optional[int]]:
        """Tracker ids when present; None per box otherwise."""
        try:
            raw = getattr(boxes, 'id', None)
            if raw is None:
                return [None] * count
            values = raw.int().cpu().tolist()
            if len(values) != count:
                return [None] * count
            return [int(value) for value in values]
        except Exception:
            return [None] * count


def class_label(class_id: int) -> str:
    """Canonical English class name for a COCO id."""
    return Config.VEHICLE_CLASS_LABELS.get(int(class_id), 'unknown')


def class_label_id(class_id: int) -> str:
    """Indonesian class name shown in the UI and stored in the database."""
    return Config.VEHICLE_CLASS_LABELS_ID.get(int(class_id), 'Lainnya')
