"""Detection data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    def center(self) -> tuple[float, float]:
        """Return the box center point."""

        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True, slots=True)
class DetectionItem:
    """Normalized object detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    """Normalized detection result for one frame."""

    frame_id: str
    timestamp: str
    source: str
    objects: list[DetectionItem]

