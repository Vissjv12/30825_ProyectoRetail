"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.alerts.manager import AlertManager
from app.config.models import InventoryConfig, SettingsConfig, ZonesConfig
from app.core.models import FramePayload
from app.detection.detector import YoloDetector
from app.inventory.engine import InventoryEngine
from app.llm.client import LlmClient
from app.llm.models import LlmRequest
from app.rules.engine import RulesEngine
from app.rules.models import RulesContext
from app.zones.validator import ZoneValidator


@dataclass
class MonitoringPipeline:
    """Coordinate all system modules without coupling them together."""

    settings: SettingsConfig
    zones: ZonesConfig
    inventory_config: InventoryConfig

    def run(self, frame: FramePayload) -> dict[str, Any]:
        detector = YoloDetector(self.settings.model_path, self.settings.confidence_threshold)
        detection_payload = detector.detect(frame)
        detection_frame = detector.parser.parse(
            frame_id=detection_payload["frame_id"],
            timestamp=detection_payload["timestamp"],
            source=detection_payload["source"],
            result=_result_from_payload(detection_payload),
        )

        inventory_summary = InventoryEngine().build_summary(detection_frame)
        zone_result = ZoneValidator(self.zones.zones).validate(detection_frame)
        rules_evaluation = RulesEngine(self.inventory_config.expectations).evaluate(
            RulesContext(inventory=inventory_summary, zone_validation=zone_result)
        )
        alert_batch = AlertManager().build_alerts(rules_evaluation)
        llm_response = LlmClient(self.settings.llm_provider).analyze(
            LlmRequest(
                summary={
                    "detections": detection_payload.get("objects", []),
                    "inventory": asdict(inventory_summary),
                    "alerts": [asdict(alert) for alert in alert_batch.alerts],
                    "events": [asdict(event) for event in rules_evaluation.events],
                }
            )
        )

        return {
            "frame": {"frame_id": frame.frame_id, "timestamp": frame.timestamp, "source": frame.source},
            "detections": detection_payload,
            "inventory": asdict(inventory_summary),
            "zones": asdict(zone_result),
            "rules": asdict(rules_evaluation),
            "alerts": asdict(alert_batch),
            "llm": asdict(llm_response),
        }


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
