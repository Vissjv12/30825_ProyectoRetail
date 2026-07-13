"""Video sampling and analysis orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import base64
import cv2
import numpy as np

from app.core.exceptions import VideoAnalysisError
from app.core.models import FramePayload
from app.llm.client import VIDEO_ALERTADOR_PROMPT
from app.orchestration.pipeline import MonitoringPipeline


@dataclass(frozen=True, slots=True)
class VideoSamplingConfig:
    """Controls how many video frames are analyzed."""

    sample_every_seconds: float = 1.0
    max_samples: int = 20


@dataclass
class VideoAnalyzer:
    """Analyze selected frames from a video file."""

    pipeline: MonitoringPipeline
    config: VideoSamplingConfig = VideoSamplingConfig()

    def analyze(self, video_path: Path, source: str) -> dict[str, Any]:
        """Sample a video, analyze frames, and return a consolidated payload."""

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoAnalysisError(f"Unable to open video file: {video_path}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(fps * self.config.sample_every_seconds))
            sampled_results: list[dict[str, Any]] = []
            frame_index = 0

            while len(sampled_results) < self.config.max_samples:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    sampled_results.append(self._analyze_frame(frame, frame_index, fps, source))
                frame_index += 1
        finally:
            capture.release()

        if not sampled_results:
            raise VideoAnalysisError(f"No frames could be sampled from video file: {video_path}")

        final_result = sampled_results[-1]
        video_summary = self._build_video_summary(source, total_frames, sampled_results)
        llm_response = self.pipeline.analyze_summary(video_summary, prompt=VIDEO_ALERTADOR_PROMPT)

        return {
            "source": source,
            "frames_total": total_frames,
            "frames_analyzed": len(sampled_results),
            "sample_every_seconds": self.config.sample_every_seconds,
            "samples": sampled_results,
            "final_result": final_result,
            "video_summary": video_summary,
            "llm": {
                "analysis": llm_response.analysis,
                "raw": llm_response.raw,
            },
        }

    def analyze_stream(self, video_path: Path, source: str):
        """Sample a video and yield results frame-by-frame, ending with the LLM analysis."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoAnalysisError(f"Unable to open video file: {video_path}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(fps * self.config.sample_every_seconds))
            sampled_results: list[dict[str, Any]] = []
            frame_index = 0

            while len(sampled_results) < self.config.max_samples:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    result = self._analyze_frame(frame, frame_index, fps, source)
                    sampled_results.append(result)
                    
                    # Annotate the frame and encode to base64
                    annotated = self._draw_annotations(frame.copy(), result)
                    success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    b64_img = base64.b64encode(buffer).decode("utf-8") if success else ""

                    # Yield the result for the current frame
                    yield {
                        "type": "frame",
                        "data": result,
                        "image_b64": b64_img,
                        "progress": {
                            "current": len(sampled_results),
                            "total": self.config.max_samples,
                            "frame_index": frame_index,
                            "video_total_frames": total_frames
                        }
                    }
                frame_index += 1
        finally:
            capture.release()

        if not sampled_results:
            raise VideoAnalysisError(f"No frames could be sampled from video file: {video_path}")

        final_result = sampled_results[-1]
        video_summary = self._build_video_summary(source, total_frames, sampled_results)
        
        # After all frames, do the final LLM analysis
        llm_response = self.pipeline.analyze_summary(video_summary, prompt=VIDEO_ALERTADOR_PROMPT)
        
        yield {
            "type": "summary",
            "data": {
                "source": source,
                "frames_total": total_frames,
                "frames_analyzed": len(sampled_results),
                "sample_every_seconds": self.config.sample_every_seconds,
                "final_result": final_result,
                "video_summary": video_summary,
                "llm": {
                    "analysis": llm_response.analysis,
                    "raw": llm_response.raw,
                },
            }
        }

    def _analyze_frame(self, frame: Any, frame_index: int, fps: float, source: str) -> dict[str, Any]:
        """Analyze one sampled video frame without per-frame LLM calls."""

        second = frame_index / fps if fps else 0.0
        payload = FramePayload(
            frame_id=f"{Path(source).name}-frame-{frame_index}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            image=frame,
            metadata={"frame_index": frame_index, "video_second": second},
        )
        result = self.pipeline.run(payload, include_llm=False)
        result["frame"]["video_second"] = second
        result["frame"]["frame_index"] = frame_index
        return result

    def _build_video_summary(
        self,
        source: str,
        total_frames: int,
        sampled_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a compact historical summary for the video-level LLM diagnosis."""

        final_result = sampled_results[-1]
        snapshots = [self._build_snapshot(index, result) for index, result in enumerate(sampled_results)]
        keyframes = self._select_keyframes(snapshots)
        event_counts = Counter(
            event["event_type"]
            for snapshot in snapshots
            for event in snapshot["events"]
            if event.get("event_type")
        )
        alert_counts = Counter(
            alert["event_type"]
            for snapshot in snapshots
            for alert in snapshot["alerts"]
            if alert.get("event_type")
        )

        return {
            "source": source,
            "summary_type": "video_history",
            "frames_total": total_frames,
            "frames_analyzed": len(sampled_results),
            "sample_every_seconds": self.config.sample_every_seconds,
            "frame_width": final_result["zones"].get("frame_width"),
            "frame_height": final_result["zones"].get("frame_height"),
            "zone_profile": final_result.get("zone_profile"),
            "initial_state": keyframes[0] if keyframes else None,
            "final_state": keyframes[-1] if keyframes else None,
            "timeline_keyframes": keyframes,
            "changes": self._build_changes(keyframes),
            "event_counts": dict(event_counts),
            "alert_counts": dict(alert_counts),
            "final_inventory": final_result["inventory"],
            "final_zones": final_result["zones"],
            "final_events": final_result["rules"]["events"],
            "final_alerts": final_result["alerts"]["alerts"],
        }

    def _build_snapshot(self, sample_index: int, result: dict[str, Any]) -> dict[str, Any]:
        """Extract the minimum useful state from one analyzed frame."""

        frame = result.get("frame", {})
        return {
            "sample_index": sample_index,
            "frame_index": frame.get("frame_index"),
            "video_second": round(float(frame.get("video_second", 0.0)), 2),
            "inventory_total": self._inventory_total(result),
            "inventory_by_zone": self._inventory_by_zone(result),
            "events": self._compact_events(result.get("rules", {}).get("events", [])),
            "alerts": self._compact_alerts(result.get("alerts", {}).get("alerts", [])),
        }

    @staticmethod
    def _inventory_total(result: dict[str, Any]) -> dict[str, int]:
        """Return total inventory counts by class for one frame."""

        totals: dict[str, int] = {}
        for item in result.get("inventory", {}).get("items", []):
            class_name = item.get("class_name")
            if class_name:
                totals[str(class_name)] = int(item.get("count", 0))
        return totals

    @staticmethod
    def _inventory_by_zone(result: dict[str, Any]) -> dict[str, dict[str, int]]:
        """Return inventory counts grouped by zone and class for one frame."""

        counts: dict[str, dict[str, int]] = {}
        for item in result.get("zones", {}).get("detections", []):
            zone_id = item.get("zone_id")
            detection = item.get("detection", {})
            class_name = detection.get("class_name")
            if not zone_id or not class_name:
                continue
            zone_counts = counts.setdefault(str(zone_id), {})
            zone_counts[str(class_name)] = zone_counts.get(str(class_name), 0) + 1
        return counts

    @staticmethod
    def _compact_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep event fields useful for video diagnosis."""

        return [
            {
                "event_type": event.get("event_type"),
                "severity": event.get("severity"),
                "zone_id": event.get("zone_id"),
                "class_name": event.get("class_name"),
                "message": event.get("message"),
                "details": event.get("details", {}),
            }
            for event in events
        ]

    @staticmethod
    def _compact_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep alert fields useful for video diagnosis."""

        return [
            {
                "event_type": alert.get("event_type"),
                "severity": alert.get("severity"),
                "zone_id": alert.get("zone_id"),
                "class_name": alert.get("class_name"),
                "message": alert.get("message"),
            }
            for alert in alerts
        ]

    def _select_keyframes(self, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return first, final, and every snapshot where business state changes."""

        keyframes: list[dict[str, Any]] = []
        previous_signature: tuple[Any, ...] | None = None
        for snapshot in snapshots:
            signature = self._state_signature(snapshot)
            if previous_signature is None or signature != previous_signature:
                keyframes.append(snapshot)
                previous_signature = signature

        if snapshots and keyframes[-1]["sample_index"] != snapshots[-1]["sample_index"]:
            keyframes.append(snapshots[-1])
        return keyframes

    @staticmethod
    def _state_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
        """Build a comparable signature of inventory, events, and alerts."""

        event_signature = sorted(
            (event.get("event_type"), event.get("zone_id"), event.get("class_name"))
            for event in snapshot["events"]
        )
        alert_signature = sorted(
            (alert.get("event_type"), alert.get("zone_id"), alert.get("class_name"))
            for alert in snapshot["alerts"]
        )
        return (
            tuple(sorted(snapshot["inventory_total"].items())),
            tuple((zone, tuple(sorted(classes.items()))) for zone, classes in sorted(snapshot["inventory_by_zone"].items())),
            tuple(event_signature),
            tuple(alert_signature),
        )

    @staticmethod
    def _build_changes(keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Describe inventory count changes between consecutive keyframes."""

        changes: list[dict[str, Any]] = []
        for previous, current in zip(keyframes, keyframes[1:]):
            zone_changes: list[dict[str, Any]] = []
            zones = set(previous["inventory_by_zone"]) | set(current["inventory_by_zone"])
            for zone_id in sorted(zones):
                previous_counts = previous["inventory_by_zone"].get(zone_id, {})
                current_counts = current["inventory_by_zone"].get(zone_id, {})
                classes = set(previous_counts) | set(current_counts)
                for class_name in sorted(classes):
                    before = previous_counts.get(class_name, 0)
                    after = current_counts.get(class_name, 0)
                    if before == after:
                        continue
                    zone_changes.append(
                        {
                            "zone_id": zone_id,
                            "class_name": class_name,
                            "from": before,
                            "to": after,
                            "delta": after - before,
                        }
                    )
            if zone_changes:
                changes.append(
                    {
                        "from_second": previous["video_second"],
                        "to_second": current["video_second"],
                        "from_sample_index": previous["sample_index"],
                        "to_sample_index": current["sample_index"],
                        "zone_changes": zone_changes,
                        "events_at_to_state": current["events"],
                        "alerts_at_to_state": current["alerts"],
                    }
                )
        return changes

    def _draw_annotations(self, frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        """Overlay bounding boxes and zones on the frame."""
        ZONE_COLORS = {
            "top":    (0, 255, 128),
            "middle": (0, 200, 255),
            "bottom": (255, 160, 0),
        }
        DEFAULT_COLOR = (180, 180, 180)

        # Draw zones
        height, width = frame.shape[:2]
        for zone in self.pipeline.zones.profile.to_pixel_zones(width, height):
            color = ZONE_COLORS.get(zone.zone_id, DEFAULT_COLOR)
            x1, y1, x2, y2 = int(zone.x1), int(zone.y1), int(zone.x2), int(zone.y2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
            frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            label = zone.name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1), (x1 + tw + 8, y1 + th + 8), color, thickness=-1)
            cv2.putText(frame, label, (x1 + 4, y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw bounding boxes
        for obj in result.get("detections", {}).get("objects", []):
            bbox = obj.get("bbox", {})
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cls_name = obj.get("class_name", "unknown")
            conf = obj.get("confidence", 0.0)
            color = (0, 0, 255) # Red bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, thickness=-1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
