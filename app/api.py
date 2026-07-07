"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from app.config.loader import load_inventory_config, load_settings_config, load_zones_config
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

    return app

