from __future__ import annotations

from app.detection.models import BoundingBox, DetectionFrame, DetectionItem
from app.inventory.engine import InventoryEngine
from app.rules.engine import RulesEngine
from app.rules.models import ExpectedItem, RulesContext, ZoneInventoryExpectation
from app.zones.models import Zone
from app.zones.validator import ZoneValidator


def test_rules_engine_detects_inventory_low() -> None:
    zones = [Zone(zone_id="A", name="Shelf A", x1=0, y1=0, x2=100, y2=100, allowed_classes=["banana"])]
    detection_frame = DetectionFrame(
        frame_id="frame-1",
        timestamp="2026-07-07T00:00:00Z",
        source="0",
        objects=[],
    )
    inventory = InventoryEngine().build_summary(detection_frame)
    zone_validation = ZoneValidator(zones).validate(detection_frame)
    engine = RulesEngine([ZoneInventoryExpectation(zone_id="A", items=[ExpectedItem("banana", min_count=1)])])
    result = engine.evaluate(RulesContext(inventory=inventory, zone_validation=zone_validation))
    assert result.events[0].event_type == "inventory_low"

