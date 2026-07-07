"""YOLO detector adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ultralytics import YOLO

from app.detection.parser import DetectionParser
from app.core.models import FramePayload

logger = logging.getLogger(__name__)


@dataclass
class YoloDetector:
    """Run Ultralytics YOLO on frames and return normalized JSON."""

    model_path: str
    conf_threshold: float = 0.25
    parser: DetectionParser = DetectionParser()

    def __post_init__(self) -> None:
        self._model = YOLO(self.model_path)
        logger.info("Loaded YOLO model: %s", self.model_path)

    def detect(self, frame: FramePayload) -> dict[str, Any]:
        """Detect objects in one frame and return a JSON-ready dict."""

        results = self._model.predict(frame.image, conf=self.conf_threshold)
        parsed = self.parser.parse(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            source=frame.source,
            result=results[0],
        )
        return self.parser.to_dict(parsed)

