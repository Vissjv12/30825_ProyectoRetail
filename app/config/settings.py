"""Settings loader for JSON-based configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class CameraSettings:
    """Camera configuration."""

    source: str | int
    timeout_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Top-level application settings."""

    camera: CameraSettings


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and validate it is an object."""

    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration file must contain a JSON object: {path}")
    return data


def load_settings(path: str | Path) -> AppSettings:
    """Load application settings from a JSON file."""

    raw = load_json(Path(path))
    camera = raw.get("camera")
    if not isinstance(camera, dict):
        raise ConfigurationError("Missing or invalid 'camera' section in settings.json")
    source = camera.get("source")
    if not isinstance(source, (str, int)):
        raise ConfigurationError("camera.source must be a string RTSP URL or an integer webcam index")
    if isinstance(source, str) and not source.strip():
        raise ConfigurationError("camera.source must be a non-empty string")
    timeout_seconds = float(camera.get("timeout_seconds", 5.0))
    return AppSettings(camera=CameraSettings(source=source, timeout_seconds=timeout_seconds))
