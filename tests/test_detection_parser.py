from __future__ import annotations

from types import SimpleNamespace

from app.detection.parser import DetectionParser


class _ArrayLike:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return self._data


def test_detection_parser_to_dict() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_ArrayLike([[1, 2, 3, 4]]),
            conf=_ArrayLike([0.9]),
            cls=_ArrayLike([0]),
            id=None,
        ),
        names={0: "person"},
    )
    parser = DetectionParser()
    parsed = parser.parse("frame-1", "2026-07-07T00:00:00Z", "0", result)
    payload = parser.to_dict(parsed)
    assert payload["frame_id"] == "frame-1"
    assert payload["objects"][0]["class_name"] == "person"
    assert payload["objects"][0]["confidence"] == 0.9

