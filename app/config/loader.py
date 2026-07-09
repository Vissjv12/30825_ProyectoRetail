"""Load JSON configuration files into typed configuration objects."""

from __future__ import annotations

from pathlib import Path

from app.config.models import InventoryConfig, LlmConfig, SettingsConfig, ZonesConfig
from app.config.settings import load_json
from app.core.exceptions import ConfigurationError
from app.rules.models import ExpectedItem, ZoneInventoryExpectation
from app.zones.models import Zone


def load_settings_config(path: str | Path) -> SettingsConfig:
    raw = load_json(Path(path))
    camera = raw.get("camera")
    model = raw.get("model")
    llm = raw.get("llm", {})
    if not isinstance(camera, dict):
        raise ConfigurationError("Missing or invalid 'camera' section in settings.json")
    if not isinstance(model, dict):
        raise ConfigurationError("Missing or invalid 'model' section in settings.json")
    if not isinstance(llm, dict):
        raise ConfigurationError("Missing or invalid 'llm' section in settings.json")

    camera_source = camera.get("source")
    if not isinstance(camera_source, (str, int)):
        raise ConfigurationError("camera.source must be a string RTSP URL or webcam index")
    model_path = model.get("path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ConfigurationError("model.path must be a non-empty string")

    return SettingsConfig(
        camera_source=camera_source,
        model_path=model_path,
        confidence_threshold=float(model.get("confidence_threshold", 0.25)),
        llm=LlmConfig(
            provider=str(llm.get("provider", "placeholder")),
            model=str(llm.get("model", "grok-4.5")),
            api_key_env=str(llm.get("api_key_env", "XAI_API_KEY")),
            base_url=str(llm.get("base_url", "https://api.x.ai/v1/responses")),
            enabled=bool(llm.get("enabled", False)),
        ),
    )


def load_zones_config(path: str | Path) -> ZonesConfig:
    raw = load_json(Path(path))
    zones_raw = raw.get("zones")
    if not isinstance(zones_raw, list):
        raise ConfigurationError("zones.json must contain a 'zones' array")

    zones: list[Zone] = []
    for zone_raw in zones_raw:
        if not isinstance(zone_raw, dict):
            raise ConfigurationError("Each zone must be a JSON object")
        zones.append(
            Zone(
                zone_id=str(zone_raw.get("zone_id")),
                name=str(zone_raw.get("name")),
                x1=float(zone_raw.get("x1")),
                y1=float(zone_raw.get("y1")),
                x2=float(zone_raw.get("x2")),
                y2=float(zone_raw.get("y2")),
                allowed_classes=[str(item) for item in zone_raw.get("allowed_classes", [])],
            )
        )
    return ZonesConfig(zones=zones)


def load_inventory_config(path: str | Path) -> InventoryConfig:
    raw = load_json(Path(path))
    expectations_raw = raw.get("expectations")
    if not isinstance(expectations_raw, list):
        raise ConfigurationError("inventory.json must contain an 'expectations' array")

    expectations: list[ZoneInventoryExpectation] = []
    for expectation_raw in expectations_raw:
        if not isinstance(expectation_raw, dict):
            raise ConfigurationError("Each expectation must be a JSON object")
        items_raw = expectation_raw.get("items", [])
        if not isinstance(items_raw, list):
            raise ConfigurationError("Expectation 'items' must be an array")
        items: list[ExpectedItem] = []
        for item_raw in items_raw:
            if not isinstance(item_raw, dict):
                raise ConfigurationError("Each expected item must be a JSON object")
            items.append(
                ExpectedItem(
                    class_name=str(item_raw.get("class_name")),
                    min_count=int(item_raw.get("min_count", 0)),
                    max_count=None if item_raw.get("max_count") is None else int(item_raw.get("max_count")),
                    target_count=None if item_raw.get("target_count") is None else int(item_raw.get("target_count")),
                )
            )
        expectations.append(ZoneInventoryExpectation(zone_id=str(expectation_raw.get("zone_id")), items=items))
    return InventoryConfig(expectations=expectations)

