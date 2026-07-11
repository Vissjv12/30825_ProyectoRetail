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

    def to_pixel_zone(self, width: int, height: int, coordinate_mode: str) -> "Zone":
        """Return this zone in pixel coordinates for a concrete frame size."""

        if coordinate_mode != "normalized":
            return self
        return Zone(
            zone_id=self.zone_id,
            name=self.name,
            x1=self.x1 * width,
            y1=self.y1 * height,
            x2=self.x2 * width,
            y2=self.y2 * height,
            allowed_classes=self.allowed_classes,
        )


@dataclass(frozen=True, slots=True)
class ZoneProfile:
    """A named calibration profile for one camera/image/video geometry."""

    profile_id: str
    name: str
    reference_width: int
    reference_height: int
    coordinate_mode: str
    zones: list[Zone] = field(default_factory=list)

    def to_pixel_zones(self, width: int, height: int) -> list[Zone]:
        """Resolve all zones to pixel coordinates for a concrete frame size."""

        return [zone.to_pixel_zone(width, height, self.coordinate_mode) for zone in self.zones]


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
    profile_id: str | None
    frame_width: int | None
    frame_height: int | None
    detections: list[ZonedDetection]

