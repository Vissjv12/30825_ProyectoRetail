"""Video sampling and analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

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
