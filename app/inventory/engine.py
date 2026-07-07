"""Inventory aggregation engine."""

from __future__ import annotations

from collections import Counter

from app.detection.models import DetectionFrame
from app.inventory.models import InventoryItemSummary, InventorySummary


class InventoryEngine:
    """Aggregate detections by class."""

    def build_summary(self, detection_frame: DetectionFrame) -> InventorySummary:
        """Count detections per class and return a summary."""

        counter = Counter(item.class_name for item in detection_frame.objects)
        items = [
            InventoryItemSummary(class_name=class_name, count=count)
            for class_name, count in sorted(counter.items(), key=lambda pair: pair[0])
        ]
        return InventorySummary(
            frame_id=detection_frame.frame_id,
            timestamp=detection_frame.timestamp,
            source=detection_frame.source,
            items=items,
        )

