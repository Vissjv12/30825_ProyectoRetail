"""Business rules engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.rules.models import (
    ExpectedItem,
    RuleEvent,
    RulesContext,
    RulesEvaluationResult,
    ZoneInventoryExpectation,
)


@dataclass
class RulesEngine:
    """Compare observed inventory against expected inventory by zone."""

    expectations: list[ZoneInventoryExpectation]

    def evaluate(self, context: RulesContext) -> RulesEvaluationResult:
        """Generate business events from inventory and zone validation."""

        events: list[RuleEvent] = []
        observed_by_zone = self._count_by_zone_and_class(context)

        for expectation in self.expectations:
            zone_counter = observed_by_zone.get(expectation.zone_id, Counter())
            for expected_item in expectation.items:
                observed_count = zone_counter.get(expected_item.class_name, 0)
                events.extend(self._build_events(expectation.zone_id, expected_item, observed_count))

        return RulesEvaluationResult(
            frame_id=context.inventory.frame_id,
            timestamp=context.inventory.timestamp,
            source=context.inventory.source,
            events=events,
        )

    def _count_by_zone_and_class(self, context: RulesContext) -> dict[str, Counter]:
        """Count detected objects by zone and class."""

        counters: dict[str, Counter] = {}
        for zoned_detection in context.zone_validation.detections:
            if zoned_detection.zone_id is None:
                continue
            counters.setdefault(zoned_detection.zone_id, Counter())[zoned_detection.detection.class_name] += 1
        return counters

    @staticmethod
    def _build_events(zone_id: str, expected_item: ExpectedItem, observed_count: int) -> list[RuleEvent]:
        """Build events for one expected item."""

        events: list[RuleEvent] = []
        if observed_count < expected_item.min_count:
            events.append(
                RuleEvent(
                    event_type="inventory_low",
                    severity="high",
                    zone_id=zone_id,
                    class_name=expected_item.class_name,
                    message=f"Inventory low for {expected_item.class_name}: expected at least {expected_item.min_count}, found {observed_count}",
                    details={"expected_min": expected_item.min_count, "observed": observed_count},
                )
            )
        if expected_item.max_count is not None and observed_count > expected_item.max_count:
            events.append(
                RuleEvent(
                    event_type="inventory_excess",
                    severity="medium",
                    zone_id=zone_id,
                    class_name=expected_item.class_name,
                    message=f"Inventory excess for {expected_item.class_name}: expected at most {expected_item.max_count}, found {observed_count}",
                    details={"expected_max": expected_item.max_count, "observed": observed_count},
                )
            )
        if expected_item.target_count is not None and observed_count != expected_item.target_count:
            events.append(
                RuleEvent(
                    event_type="inventory_mismatch",
                    severity="medium",
                    zone_id=zone_id,
                    class_name=expected_item.class_name,
                    message=f"Inventory mismatch for {expected_item.class_name}: expected {expected_item.target_count}, found {observed_count}",
                    details={"expected_target": expected_item.target_count, "observed": observed_count},
                )
            )
        return events

