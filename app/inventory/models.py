"""Inventory models and summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.detection.models import DetectionFrame


@dataclass(frozen=True, slots=True)
class InventoryItemSummary:
    """Count of detected objects for a given class."""

    class_name: str
    count: int


@dataclass(frozen=True, slots=True)
class InventorySummary:
    """Aggregated inventory snapshot for one frame."""

    frame_id: str
    timestamp: str
    source: str
    items: list[InventoryItemSummary] = field(default_factory=list)

