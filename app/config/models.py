"""Configuration models loaded from JSON files."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.rules.models import ZoneInventoryExpectation
from app.zones.models import ZoneProfile


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """LLM provider settings used for JSON-only diagnosis."""

    provider: str = "placeholder"
    model: str = "grok-4.5"
    api_key_env: str = "XAI_API_KEY"
    base_url: str = "https://api.x.ai/v1/responses"
    enabled: bool = False


@dataclass(frozen=True, slots=True)
class SettingsConfig:
    """Runtime settings used by the application."""

    camera_source: str | int
    model_path: str
    confidence_threshold: float = 0.25
    llm: LlmConfig = field(default_factory=LlmConfig)


@dataclass(frozen=True, slots=True)
class ZonesConfig:
    """Physical zone definitions."""

    active_profile: str
    profile: ZoneProfile


@dataclass(frozen=True, slots=True)
class InventoryConfig:
    """Expected inventory definitions."""

    expectations: list[ZoneInventoryExpectation] = field(default_factory=list)

