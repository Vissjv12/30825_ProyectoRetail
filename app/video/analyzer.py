"""Video sampling and analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import base64
import cv2
import numpy as np

from app.core.exceptions import VideoAnalysisError
from app.core.models import FramePayload
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
        video_summary = {
            "source": source,
            "frames_total": total_frames,
            "frames_analyzed": len(sampled_results),
            "sample_every_seconds": self.config.sample_every_seconds,
            "final_inventory": final_result["inventory"],
            "final_zones": final_result["zones"],
            "final_events": final_result["rules"]["events"],
            "final_alerts": final_result["alerts"]["alerts"],
        }
        llm_response = self.pipeline.analyze_summary(video_summary)

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
        video_summary = {
            "source": source,
            "frames_total": total_frames,
            "frames_analyzed": len(sampled_results),
            "sample_every_seconds": self.config.sample_every_seconds,
            "final_inventory": final_result["inventory"],
            "final_zones": final_result["zones"],
            "final_events": final_result["rules"]["events"],
            "final_alerts": final_result["alerts"]["alerts"],
        }
        
        # After all frames, do the final LLM analysis
        llm_response = self.pipeline.analyze_summary(video_summary)
        
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
        return result

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
