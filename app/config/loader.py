"""Load JSON configuration files into typed configuration objects."""

from __future__ import annotations

from pathlib import Path

from app.config.models import InventoryConfig, LlmConfig, SettingsConfig, ZonesConfig
from app.config.settings import load_json
from app.core.exceptions import ConfigurationError
from app.rules.models import ExpectedItem, ZoneInventoryExpectation
from app.zones.models import Zone, ZoneProfile


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


def load_zones_config(path: str | Path, profile_id: str | None = None) -> ZonesConfig:
    raw = load_json(Path(path))
    if "profiles" in raw:
        return _load_profiled_zones_config(raw, profile_id)

    zones_raw = raw.get("zones")
    if not isinstance(zones_raw, list):
        raise ConfigurationError("zones.json must contain a 'zones' array or a 'profiles' object")

    profile = ZoneProfile(
        profile_id="legacy",
        name="Legacy zones",
        reference_width=0,
        reference_height=0,
        coordinate_mode="pixels",
        zones=[_parse_zone(zone_raw) for zone_raw in zones_raw],
    )
    return ZonesConfig(active_profile="legacy", profile=profile)


def _load_profiled_zones_config(raw: dict, profile_id: str | None) -> ZonesConfig:
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ConfigurationError("zones.json 'profiles' must be a non-empty object")

    active_profile = str(raw.get("active_profile") or next(iter(profiles_raw)))
    selected_profile = profile_id or active_profile
    profile_raw = profiles_raw.get(selected_profile)
    if not isinstance(profile_raw, dict):
        raise ConfigurationError(f"Zone profile not found: {selected_profile}")

    profile = _parse_zone_profile(selected_profile, profile_raw)
    return ZonesConfig(active_profile=selected_profile, profile=profile)


def _parse_zone_profile(profile_id: str, profile_raw: dict) -> ZoneProfile:
    zones_raw = profile_raw.get("zones")
    if not isinstance(zones_raw, list):
        raise ConfigurationError(f"Zone profile '{profile_id}' must contain a zones array")

    coordinate_mode = str(profile_raw.get("coordinate_mode", "normalized"))
    if coordinate_mode not in {"normalized", "pixels"}:
        raise ConfigurationError("Zone coordinate_mode must be 'normalized' or 'pixels'")

    return ZoneProfile(
        profile_id=profile_id,
        name=str(profile_raw.get("name", profile_id)),
        reference_width=int(profile_raw.get("reference_width", 0)),
        reference_height=int(profile_raw.get("reference_height", 0)),
        coordinate_mode=coordinate_mode,
        zones=[_parse_zone(zone_raw) for zone_raw in zones_raw],
    )


def _parse_zone(zone_raw: object) -> Zone:
    if not isinstance(zone_raw, dict):
        raise ConfigurationError("Each zone must be a JSON object")
    return Zone(
        zone_id=str(zone_raw.get("zone_id")),
        name=str(zone_raw.get("name")),
        x1=float(zone_raw.get("x1")),
        y1=float(zone_raw.get("y1")),
        x2=float(zone_raw.get("x2")),
        y2=float(zone_raw.get("y2")),
        allowed_classes=[str(item) for item in zone_raw.get("allowed_classes", [])],
    )


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

