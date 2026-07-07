from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config.settings import CameraSettings
from app.core.exceptions import CameraError
from app.infrastructure.camera_manager import CameraManager


@dataclass
class _DummyCapture:
    opened: bool = True

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def read(self):
        return True, "frame"

    def release(self) -> None:
        return None


def test_camera_manager_read_raises_without_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.infrastructure.camera_manager as camera_module

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda source: _DummyCapture())
    camera = CameraManager(CameraSettings(source="0"))
    payload = camera.read()
    assert payload.source == "0"
    assert payload.image == "frame"
    camera.close()


def test_camera_manager_raises_when_camera_cannot_open(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.infrastructure.camera_manager as camera_module

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda source: _DummyCapture(opened=False))
    with pytest.raises(CameraError):
        CameraManager(CameraSettings(source="bad-source"))

