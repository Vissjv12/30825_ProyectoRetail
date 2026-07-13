"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.alerts.manager import AlertManager
from app.config.models import InventoryConfig, SettingsConfig, ZonesConfig
from app.core.models import FramePayload
from app.detection.detector import YoloDetector
from app.inventory.engine import InventoryEngine
from app.llm.client import LlmClient
from app.llm.models import LlmRequest, LlmResponse
from app.rules.engine import RulesEngine
from app.rules.models import RulesContext
from app.zones.validator import ZoneValidator


@dataclass
class MonitoringPipeline:
    """Coordinate all system modules without coupling them together."""

    settings: SettingsConfig
    zones: ZonesConfig
    inventory_config: InventoryConfig
    _detector: YoloDetector | None = field(default=None, init=False, repr=False)

    def run(self, frame: FramePayload, include_llm: bool = True) -> dict[str, Any]:
        detector = self._get_detector()
        detection_payload = detector.detect(frame)
        detection_frame = detector.parser.parse(
            frame_id=detection_payload["frame_id"],
            timestamp=detection_payload["timestamp"],
            source=detection_payload["source"],
            result=_result_from_payload(detection_payload),
        )

        inventory_summary = InventoryEngine().build_summary(detection_frame)
        frame_width, frame_height = _frame_size(frame.image)
        pixel_zones = self.zones.profile.to_pixel_zones(frame_width, frame_height)
        zone_result = ZoneValidator(pixel_zones).validate(
            detection_frame,
            profile_id=self.zones.profile.profile_id,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        rules_evaluation = RulesEngine(self.inventory_config.expectations).evaluate(
            RulesContext(inventory=inventory_summary, zone_validation=zone_result)
        )
        alert_batch = AlertManager().build_alerts(rules_evaluation)
        llm_summary = {
            "detections": detection_payload.get("objects", []),
            "inventory": asdict(inventory_summary),
            "zones": asdict(zone_result),
            "alerts": [asdict(alert) for alert in alert_batch.alerts],
            "events": [asdict(event) for event in rules_evaluation.events],
        }
        llm_response = (
            self.analyze_summary(llm_summary)
            if include_llm
            else LlmResponse(analysis="LLM diagnosis skipped for sampled frame.", raw={"reason": "sampled-frame"})
        )

        return {
            "frame": {"frame_id": frame.frame_id, "timestamp": frame.timestamp, "source": frame.source},
            "zone_profile": {
                "profile_id": self.zones.profile.profile_id,
                "name": self.zones.profile.name,
                "coordinate_mode": self.zones.profile.coordinate_mode,
                "reference_width": self.zones.profile.reference_width,
                "reference_height": self.zones.profile.reference_height,
            },
            "detections": detection_payload,
            "inventory": asdict(inventory_summary),
            "zones": asdict(zone_result),
            "rules": asdict(rules_evaluation),
            "alerts": asdict(alert_batch),
            "llm": asdict(llm_response),
        }

    def analyze_summary(self, summary: dict[str, object], prompt: str | None = None) -> Any:
        """Generate one LLM diagnosis from a JSON-only system summary."""

        return LlmClient(self.settings.llm).analyze(LlmRequest(summary=summary, prompt=prompt))

    def _get_detector(self) -> YoloDetector:
        """Load YOLO on first use and reuse it for later frames."""

        if self._detector is None:
            self._detector = YoloDetector(self.settings.model_path, self.settings.confidence_threshold)
        return self._detector


def _result_from_payload(payload: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    objects = payload.get("objects", [])
    boxes = SimpleNamespace(
        xyxy=[[obj["bbox"]["x1"], obj["bbox"]["y1"], obj["bbox"]["x2"], obj["bbox"]["y2"]] for obj in objects],
        conf=[obj["confidence"] for obj in objects],
        cls=[obj["class_id"] for obj in objects],
        id=None,
    )
    return SimpleNamespace(boxes=boxes, names={obj["class_id"]: obj["class_name"] for obj in objects})


def _frame_size(image: Any) -> tuple[int, int]:
    """Return frame width and height from an image-like object."""

    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[0]), int(size[1])
    return 0, 0
