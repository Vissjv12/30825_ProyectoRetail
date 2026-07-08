"""FastAPI application factory."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.config.loader import load_inventory_config, load_settings_config, load_zones_config
from app.core.models import FramePayload
from app.infrastructure.camera_manager import CameraManager
from app.orchestration.pipeline import MonitoringPipeline


def create_app() -> FastAPI:
    app = FastAPI(title="Retail Monitoring System", version="0.1.0")
    settings = load_settings_config(Path("settings.json"))
    zones = load_zones_config(Path("zones.json"))
    inventory = load_inventory_config(Path("inventory.json"))
    pipeline = MonitoringPipeline(settings=settings, zones=zones, inventory_config=inventory)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run-once")
    def run_once() -> dict:
        camera = CameraManager.from_source(settings.camera_source)
        try:
            frame = camera.read()
            return pipeline.run(frame)
        finally:
            camera.close()

    @app.post("/analyze-image")
    async def analyze_image(file: UploadFile = File(...)) -> dict:
        if file.content_type is not None and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        try:
            raw = await file.read()
            image = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        frame = FramePayload(
            frame_id=file.filename or "uploaded-image",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=file.filename or "uploaded-image",
            image=np.array(image),
            metadata={"content_type": file.content_type, "filename": file.filename},
        )
        return pipeline.run(frame)

    return app
