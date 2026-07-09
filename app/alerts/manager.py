"""Alert manager that converts rule events into alerts."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.alerts.models import Alert, AlertBatch
from app.rules.models import RulesEvaluationResult


@dataclass
class AlertManager:
    """Create alerts from rule events without external side effects."""

    def build_alerts(self, evaluation: RulesEvaluationResult) -> AlertBatch:
        """Transform rule events into alert objects."""

        alerts = [
            Alert(
                alert_id=str(uuid4()),
                severity=event.severity,
                title=self._title_for_event(event.event_type),
                message=event.message,
                zone_id=event.zone_id,
                class_name=event.class_name,
                event_type=event.event_type,
                details=event.details,
            )
            for event in evaluation.events
        ]
        return AlertBatch(
            frame_id=evaluation.frame_id,
            timestamp=evaluation.timestamp,
            source=evaluation.source,
            alerts=alerts,
        )

    @staticmethod
    def _title_for_event(event_type: str) -> str:
        """Map event type to a short human-readable title."""

        mapping = {
            "inventory_absent": "Inventory absent",
            "inventory_low": "Inventory low",
            "inventory_excess": "Inventory excess",
            "inventory_mismatch": "Inventory mismatch",
            "object_misplaced": "Object misplaced",
        }
        return mapping.get(event_type, "System alert")

