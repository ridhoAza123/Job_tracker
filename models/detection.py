"""ORM model for a single speed measurement."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from db import db


class Detection(db.Model):
    __tablename__ = 'detections'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Kept for compatibility with the historical schema.
    vehicle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    vehicle_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    speed: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    camera_name: Mapped[str] = mapped_column(String(128), nullable=True)
    stream_url: Mapped[str] = mapped_column(String(512), nullable=True)
    snapshot: Mapped[str] = mapped_column(String(255), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False, index=True
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'track_id': self.track_id,
            'vehicle_type': self.vehicle_type,
            'speed': round(float(self.speed or 0.0), 2),
            'status': self.status,
            'camera_name': self.camera_name,
            'stream_url': self.stream_url,
            'snapshot': self.snapshot,
            'confidence': round(float(self.confidence), 2) if self.confidence else None,
            'timestamp': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f'<Detection {self.id} {self.vehicle_type} {self.speed:.1f}km/h>'
