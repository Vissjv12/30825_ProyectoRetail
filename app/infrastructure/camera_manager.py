"""Camera capture adapter that only knows how to read frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from app.core.exceptions import CameraError
from app.core.interfaces import IFrameSource
from app.core.models import FramePayload
from app.config.settings import CameraSettings

logger = logging.getLogger(__name__)


@dataclass
class CameraManager(IFrameSource):
    """Capture frames from a webcam or RTSP/IP camera."""

    settings: CameraSettings

    def __post_init__(self) -> None:
        self._capture = cv2.VideoCapture(self.settings.source)
        if not self._capture.isOpened():
            raise CameraError(f"Unable to open camera source: {self.settings.source}")
        logger.info("Camera source opened: %s", self.settings.source)

    def read(self) -> FramePayload:
        """Capture a single frame and return its metadata."""

        ok, frame = self._capture.read()
        if not ok:
            raise CameraError(f"Unable to read frame from: {self.settings.source}")

        frame_id = self._build_frame_id()
        logger.debug("Captured frame %s from %s", frame_id, self.settings.source)
        return FramePayload(
            frame_id=frame_id,
            timestamp=FramePayload.now_iso(),
            source=self.settings.source,
            image=frame,
        )

    @classmethod
    def from_source(cls, source: str | int, timeout_seconds: float = 5.0) -> "CameraManager":
        """Build a camera manager from primitive values."""

        return cls(CameraSettings(source=source, timeout_seconds=timeout_seconds))

    def close(self) -> None:
        """Release the camera resource."""

        if hasattr(self, "_capture") and self._capture is not None:
            self._capture.release()
            logger.info("Camera source released: %s", self.settings.source)

    def _build_frame_id(self) -> str:
        """Generate a stable frame identifier."""

        source = self.settings.source
        if isinstance(source, int):
            return f"webcam-{source}"
        return Path(source).name if "://" not in source else source
