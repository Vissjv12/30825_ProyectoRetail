from __future__ import annotations

from app.alerts.manager import AlertManager
from app.rules.models import RuleEvent, RulesEvaluationResult


def test_alert_manager_converts_events_to_alerts() -> None:
    evaluation = RulesEvaluationResult(
        frame_id="frame-1",
        timestamp="2026-07-07T00:00:00Z",
        source="0",
        events=[RuleEvent("inventory_low", "high", "A", "banana", "low", {})],
    )
    batch = AlertManager().build_alerts(evaluation)
    assert batch.alerts[0].title == "Inventory low"
    assert batch.alerts[0].severity == "high"

