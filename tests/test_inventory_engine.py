from __future__ import annotations

from app.detection.models import BoundingBox, DetectionFrame, DetectionItem
from app.inventory.engine import InventoryEngine


def test_inventory_summary_counts_by_class() -> None:
    detection_frame = DetectionFrame(
        frame_id="frame-1",
        timestamp="2026-07-07T00:00:00Z",
        source="0",
        objects=[
            DetectionItem(0, "banana", 0.9, BoundingBox(1, 1, 2, 2)),
            DetectionItem(0, "banana", 0.8, BoundingBox(3, 3, 4, 4)),
            DetectionItem(1, "apple", 0.7, BoundingBox(5, 5, 6, 6)),
        ],
    )
    summary = InventoryEngine().build_summary(detection_frame)
    assert summary.items[0].class_name == "apple"
    assert summary.items[0].count == 1
    assert summary.items[1].class_name == "banana"
    assert summary.items[1].count == 2

