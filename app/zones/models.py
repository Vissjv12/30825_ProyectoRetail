"""Zone validation models."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.detection.models import BoundingBox, DetectionItem


@dataclass(frozen=True, slots=True)
class Zone:
    """Rectangular physical area in image coordinates."""

    zone_id: str
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    allowed_classes: list[str] = field(default_factory=list)

    def contains_point(self, x: float, y: float) -> bool:
        """Return True when a point is inside the zone bounds."""

        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True, slots=True)
class ZonedDetection:
    """Detection enriched with zone information."""

    detection: DetectionItem
    zone_id: str | None
    zone_name: str | None
    is_in_allowed_zone: bool


@dataclass(frozen=True, slots=True)
class ZoneValidationResult:
    """Result of validating all detections against zones."""

    frame_id: str
    timestamp: str
    source: str
    detections: list[ZonedDetection]

