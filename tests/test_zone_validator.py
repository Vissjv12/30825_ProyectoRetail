from __future__ import annotations

from app.detection.models import BoundingBox, DetectionFrame, DetectionItem
from app.zones.models import Zone
from app.zones.validator import ZoneValidator


def test_zone_validator_assigns_zone_by_center() -> None:
    zones = [
        Zone(zone_id="A", name="Shelf A", x1=0, y1=0, x2=100, y2=100, allowed_classes=["banana", "apple"]),
        Zone(zone_id="B", name="Shelf B", x1=101, y1=0, x2=200, y2=100, allowed_classes=["bottle"]),
    ]
    detection_frame = DetectionFrame(
        frame_id="frame-1",
        timestamp="2026-07-07T00:00:00Z",
        source="0",
        objects=[
            DetectionItem(0, "banana", 0.9, BoundingBox(10, 10, 20, 20)),
            DetectionItem(1, "bottle", 0.9, BoundingBox(120, 10, 140, 20)),
        ],
    )

    result = ZoneValidator(zones).validate(detection_frame)
    assert result.detections[0].zone_id == "A"
    assert result.detections[0].is_in_allowed_zone is True
    assert result.detections[1].zone_id == "B"
    assert result.detections[1].is_in_allowed_zone is True


def test_zone_validator_marks_detection_outside_zones() -> None:
    zones = [Zone(zone_id="A", name="Shelf A", x1=0, y1=0, x2=100, y2=100)]
    detection_frame = DetectionFrame(
        frame_id="frame-1",
        timestamp="2026-07-07T00:00:00Z",
        source="0",
        objects=[DetectionItem(0, "banana", 0.9, BoundingBox(200, 200, 220, 220))],
    )

    result = ZoneValidator(zones).validate(detection_frame)
    assert result.detections[0].zone_id is None
    assert result.detections[0].is_in_allowed_zone is False

