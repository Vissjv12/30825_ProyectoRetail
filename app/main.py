"""Application entrypoint for camera capture smoke testing."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import load_settings
from app.infrastructure.camera_manager import CameraManager


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings(Path("settings.json"))
    camera = CameraManager(settings.camera)
    try:
        frame = camera.read()
        print({
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "source": frame.source,
        })
    finally:
        camera.close()


if __name__ == "__main__":
    main()

