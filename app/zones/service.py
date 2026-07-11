"""Read and write zone calibration profiles."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from app.config.loader import load_zones_config
from app.config.settings import load_json
from app.core.exceptions import ConfigurationError


class ZoneProfileService:
    """Manage persisted zone profiles in zones.json."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_all(self) -> dict[str, Any]:
        """Return the full zones configuration."""

        return load_json(self.path)

    def read_profile(self, profile_id: str) -> dict[str, Any]:
        """Return one zone profile by id."""

        raw = self.read_all()
        profiles = self._profiles(raw)
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict):
            raise ConfigurationError(f"Zone profile not found: {profile_id}")
        return {"profile_id": profile_id, **profile}

    def set_active_profile(self, profile_id: str) -> dict[str, Any]:
        """Set the active profile after confirming it exists."""

        raw = self.read_all()
        profiles = self._profiles(raw)
        if profile_id not in profiles:
            raise ConfigurationError(f"Zone profile not found: {profile_id}")
        raw["active_profile"] = profile_id
        self._write(raw)
        return raw

    def save_profile(self, profile_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        """Create or replace one zone profile and persist zones.json."""

        raw = self._ensure_profiled_config(self.read_all())
        profiles = self._profiles(raw)
        sanitized_profile = self._sanitize_profile(profile)
        profiles[profile_id] = sanitized_profile
        raw["profiles"] = profiles
        raw.setdefault("active_profile", profile_id)
        self._validate_raw(raw, profile_id)
        self._write(raw)
        return {"profile_id": profile_id, **sanitized_profile}

    def ensure_resolution_profile(
        self,
        media_type: str,
        width: int,
        height: int,
        base_profile_id: str | None = None,
    ) -> dict[str, Any]:
        """Return an existing profile for a resolution or create one from a base profile."""

        if width <= 0 or height <= 0:
            raise ConfigurationError("width and height must be positive integers")

        raw = self._ensure_profiled_config(self.read_all())
        profiles = self._profiles(raw)
        profile_id = self._resolution_profile_id(media_type, width, height)
        matching_profile_id = self._find_resolution_profile(profiles, media_type, width, height)
        if matching_profile_id is not None:
            return {"profile_id": matching_profile_id, "created": False, **profiles[matching_profile_id]}

        existing = profiles.get(profile_id)
        if isinstance(existing, dict):
            return {"profile_id": profile_id, "created": False, **existing}

        active_profile_id = str(raw.get("active_profile") or next(iter(profiles)))
        source_profile_id = base_profile_id or active_profile_id
        source_profile = profiles.get(source_profile_id)
        if not isinstance(source_profile, dict):
            raise ConfigurationError(f"Base zone profile not found: {source_profile_id}")

        normalized_zones = self._zones_as_normalized(source_profile)
        profile = {
            "name": f"{media_type.title()} {width}x{height}",
            "reference_width": width,
            "reference_height": height,
            "coordinate_mode": "normalized",
            "zones": normalized_zones,
        }
        profiles[profile_id] = profile
        raw["profiles"] = profiles
        self._validate_raw(raw, profile_id)
        self._write(raw)
        return {"profile_id": profile_id, "created": True, **profile}

    @staticmethod
    def _resolution_profile_id(media_type: str, width: int, height: int) -> str:
        clean_type = ZoneProfileService._clean_media_type(media_type)
        if not clean_type:
            clean_type = "media"
        return f"{clean_type}_{width}x{height}"

    @staticmethod
    def _clean_media_type(media_type: str) -> str:
        return "".join(ch for ch in media_type.lower() if ch.isalnum() or ch in {"_", "-"}).strip("_-")

    @staticmethod
    def _find_resolution_profile(
        profiles: dict[str, Any],
        media_type: str,
        width: int,
        height: int,
    ) -> str | None:
        clean_type = ZoneProfileService._clean_media_type(media_type)
        for profile_id, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            if clean_type and not profile_id.startswith(clean_type):
                continue
            if int(profile.get("reference_width", 0)) == width and int(profile.get("reference_height", 0)) == height:
                return profile_id
        return None

    @staticmethod
    def _zones_as_normalized(profile: dict[str, Any]) -> list[dict[str, Any]]:
        zones = profile.get("zones")
        if not isinstance(zones, list):
            raise ConfigurationError("Base zone profile must contain a zones array")
        mode = str(profile.get("coordinate_mode", "normalized"))
        reference_width = int(profile.get("reference_width", 0))
        reference_height = int(profile.get("reference_height", 0))

        normalized: list[dict[str, Any]] = []
        for zone in zones:
            if not isinstance(zone, dict):
                raise ConfigurationError("Each zone must be a JSON object")
            item = dict(zone)
            if mode == "pixels":
                if reference_width <= 0 or reference_height <= 0:
                    raise ConfigurationError("Pixel profiles need reference_width and reference_height")
                item["x1"] = float(zone.get("x1")) / reference_width
                item["x2"] = float(zone.get("x2")) / reference_width
                item["y1"] = float(zone.get("y1")) / reference_height
                item["y2"] = float(zone.get("y2")) / reference_height
            normalized.append(item)
        return normalized

    def _ensure_profiled_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert the legacy zones array into a single profile in memory."""

        if "profiles" in raw:
            return raw
        zones = raw.get("zones")
        if not isinstance(zones, list):
            raise ConfigurationError("zones.json must contain zones or profiles")
        return {
            "active_profile": "legacy",
            "profiles": {
                "legacy": {
                    "name": "Legacy zones",
                    "reference_width": 0,
                    "reference_height": 0,
                    "coordinate_mode": "pixels",
                    "zones": zones,
                }
            },
        }

    @staticmethod
    def _profiles(raw: dict[str, Any]) -> dict[str, Any]:
        profiles = raw.get("profiles")
        if not isinstance(profiles, dict):
            raise ConfigurationError("zones.json must contain a 'profiles' object")
        return profiles

    @staticmethod
    def _sanitize_profile(profile: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(profile, dict):
            raise ConfigurationError("Zone profile payload must be a JSON object")
        zones = profile.get("zones")
        if not isinstance(zones, list):
            raise ConfigurationError("Zone profile must contain a zones array")
        coordinate_mode = str(profile.get("coordinate_mode", "normalized"))
        if coordinate_mode not in {"normalized", "pixels"}:
            raise ConfigurationError("coordinate_mode must be 'normalized' or 'pixels'")
        return {
            "name": str(profile.get("name", "Zone profile")),
            "reference_width": int(profile.get("reference_width", 0)),
            "reference_height": int(profile.get("reference_height", 0)),
            "coordinate_mode": coordinate_mode,
            "zones": zones,
        }

    def _validate_raw(self, raw: dict[str, Any], profile_id: str) -> None:
        """Validate by reusing the typed loader."""

        temp_path = self.path.with_suffix(".validation.json")
        try:
            temp_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            load_zones_config(temp_path, profile_id)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _write(self, raw: dict[str, Any]) -> None:
        """Atomically write zones.json."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=self.path.parent, encoding="utf-8") as temp_file:
            json.dump(raw, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
            temp_name = temp_file.name
        Path(temp_name).replace(self.path)
