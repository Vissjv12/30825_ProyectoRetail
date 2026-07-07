"""Parser that converts Ultralytics results into the project's JSON model."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.detection.models import BoundingBox, DetectionFrame, DetectionItem


class DetectionParser:
    """Convert raw YOLO outputs into normalized detection objects."""

    def parse(self, frame_id: str, timestamp: str, source: str, result: Any) -> DetectionFrame:
        """Normalize one Ultralytics result object into DetectionFrame."""

        objects: list[DetectionItem] = []
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}
        if boxes is None:
            return DetectionFrame(frame_id=frame_id, timestamp=timestamp, source=source, objects=objects)

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        ids = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None and hasattr(boxes.id, "cpu") else None

        for index, (coords, confidence, class_id) in enumerate(zip(xyxy, confs, classes)):
            track_id = int(ids[index]) if ids is not None else None
            class_index = int(class_id)
            objects.append(
                DetectionItem(
                    class_id=class_index,
                    class_name=str(names.get(class_index, class_index)),
                    confidence=float(confidence),
                    bbox=BoundingBox(
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                    ),
                    track_id=track_id,
                )
            )

        return DetectionFrame(frame_id=frame_id, timestamp=timestamp, source=source, objects=objects)

    @staticmethod
    def to_dict(detection_frame: DetectionFrame) -> dict[str, Any]:
        """Serialize a detection frame to a JSON-compatible dictionary."""

        return asdict(detection_frame)

