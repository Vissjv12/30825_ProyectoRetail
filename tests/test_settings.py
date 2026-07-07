from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config.settings import load_settings
from app.core.exceptions import ConfigurationError


def test_load_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"camera": {"source": 0, "timeout_seconds": 3}}), encoding="utf-8")
    settings = load_settings(path)
    assert settings.camera.source == 0


def test_load_settings_missing_camera(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(path)
