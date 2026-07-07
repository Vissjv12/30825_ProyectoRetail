from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core.models import FramePayload
from app.detection.detector import YoloDetector


@dataclass
class _FakeModel:
    def predict(self, image, conf=0.25):
        from types import SimpleNamespace

        class _ArrayLike:
            def __init__(self, data):
                self._data = data

            def cpu(self):
                return self

            def numpy(self):
                return self._data

        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=_ArrayLike([[10, 20, 30, 40]]),
                    conf=_ArrayLike([0.8]),
                    cls=_ArrayLike([0]),
                    id=None,
                ),
                names={0: "banana"},
            )
        ]


def test_detector_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.detection.detector as detector_module

    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: _FakeModel())
    detector = YoloDetector("fake.pt")
    payload = detector.detect(FramePayload("f-1", "2026-07-07T00:00:00Z", "0", image="frame"))
    assert payload["frame_id"] == "f-1"
    assert payload["objects"][0]["class_name"] == "banana"

