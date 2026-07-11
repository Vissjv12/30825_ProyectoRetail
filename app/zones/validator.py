"""Geometry-based zone validator."""

from __future__ import annotations

from dataclasses import dataclass

from app.detection.models import DetectionFrame
from app.zones.models import Zone, ZonedDetection, ZoneValidationResult


@dataclass
class ZoneValidator:
    """Assign detections to zones using the center point of each bounding box."""

    zones: list[Zone]

    def validate(
        self,
        detection_frame: DetectionFrame,
        profile_id: str | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> ZoneValidationResult:
        """Validate all detections and return the enriched payload."""

        zoned_detections: list[ZonedDetection] = []
        for detection in detection_frame.objects:
            center_x, center_y = detection.bbox.center()
            matched_zone = self._find_zone_for_point(center_x, center_y)
            is_allowed = self._is_allowed_detection(detection.class_name, matched_zone)
            zoned_detections.append(
                ZonedDetection(
                    detection=detection,
                    zone_id=matched_zone.zone_id if matched_zone else None,
                    zone_name=matched_zone.name if matched_zone else None,
                    is_in_allowed_zone=is_allowed,
                )
            )

        return ZoneValidationResult(
            frame_id=detection_frame.frame_id,
            timestamp=detection_frame.timestamp,
            source=detection_frame.source,
            profile_id=profile_id,
            frame_width=frame_width,
            frame_height=frame_height,
            detections=zoned_detections,
        )

    def _find_zone_for_point(self, x: float, y: float) -> Zone | None:
        """Find the first zone containing the point."""

        for zone in self.zones:
            if zone.contains_point(x, y):
                return zone
        return None

    @staticmethod
    def _is_allowed_detection(class_name: str, zone: Zone | None) -> bool:
        """Check whether a detection is allowed in a zone."""

        if zone is None:
            return False
        if not zone.allowed_classes:
            return True
        return class_name in zone.allowed_classes

