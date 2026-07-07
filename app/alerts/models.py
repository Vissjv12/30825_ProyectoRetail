"""Alert models."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rules.models import RuleEvent


@dataclass(frozen=True, slots=True)
class Alert:
    """Notification-ready alert derived from a rule event."""

    alert_id: str
    severity: str
    title: str
    message: str
    zone_id: str | None
    class_name: str | None
    event_type: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AlertBatch:
    """Group of alerts for one frame."""

    frame_id: str
    timestamp: str
    source: str
    alerts: list[Alert] = field(default_factory=list)

