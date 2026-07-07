"""Configuration models loaded from JSON files."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rules.models import ZoneInventoryExpectation
from app.zones.models import Zone


@dataclass(frozen=True, slots=True)
class SettingsConfig:
    """Runtime settings used by the application."""

    camera_source: str | int
    model_path: str
    confidence_threshold: float = 0.25
    llm_provider: str = "openai"


@dataclass(frozen=True, slots=True)
class ZonesConfig:
    """Physical zone definitions."""

    zones: list[Zone] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InventoryConfig:
    """Expected inventory definitions."""

    expectations: list[ZoneInventoryExpectation] = field(default_factory=list)

