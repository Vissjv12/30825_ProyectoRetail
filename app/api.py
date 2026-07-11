"""FastAPI application factory."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import Response, StreamingResponse
# pyrefly: ignore [missing-import]
from PIL import Image

from app.config.loader import load_inventory_config, load_settings_config, load_zones_config
from app.core.exceptions import ConfigurationError, VideoAnalysisError
from app.core.models import FramePayload
from app.infrastructure.camera_manager import CameraManager
from app.orchestration.pipeline import MonitoringPipeline
from app.video.analyzer import VideoAnalyzer, VideoSamplingConfig
from app.zones.service import ZoneProfileService

logger = logging.getLogger(__name__)


def build_pipeline(zone_profile: str | None = None) -> MonitoringPipeline:
    """Read the latest configuration from disk and build a new pipeline."""
    settings = load_settings_config(Path("settings.json"))
    zones = load_zones_config(Path("zones.json"), zone_profile)
    inventory = load_inventory_config(Path("inventory.json"))
    return MonitoringPipeline(settings=settings, zones=zones, inventory_config=inventory)


def create_app() -> FastAPI:
    app = FastAPI(title="Retail Monitoring System", version="0.1.0")

    # Allow the HTML dashboard (any origin) to call the API from a browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──────────────────────────────────────────────────────────────
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/zones")
    def get_zones() -> dict:
        """Return the complete zone profile configuration."""

        return ZoneProfileService(Path("zones.json")).read_all()

    @app.get("/api/zones/active")
    def get_active_zone_profile() -> dict:
        """Return the currently active zone profile."""

        service = ZoneProfileService(Path("zones.json"))
        raw = service.read_all()
        active_profile = str(raw.get("active_profile", ""))
        try:
            return service.read_profile(active_profile)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/zones/active")
    def set_active_zone_profile(payload: dict = Body(...)) -> dict:
        """Set the active zone profile."""

        profile_id = payload.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise HTTPException(status_code=400, detail="profile_id is required")
        try:
            return ZoneProfileService(Path("zones.json")).set_active_profile(profile_id)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/zones/ensure-profile")
    def ensure_zone_profile(payload: dict = Body(...)) -> dict:
        """Create or reuse a normalized profile for a concrete media resolution."""

        try:
            media_type = str(payload.get("media_type", "media"))
            width = int(payload.get("width", 0))
            height = int(payload.get("height", 0))
            base_profile_id = payload.get("base_profile_id")
            if base_profile_id is not None and not isinstance(base_profile_id, str):
                raise ConfigurationError("base_profile_id must be a string")
            return ZoneProfileService(Path("zones.json")).ensure_resolution_profile(
                media_type=media_type,
                width=width,
                height=height,
                base_profile_id=base_profile_id,
            )
        except (ConfigurationError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/zones/{profile_id}")
    def get_zone_profile(profile_id: str) -> dict:
        """Return one zone profile."""

        try:
            return ZoneProfileService(Path("zones.json")).read_profile(profile_id)
        except ConfigurationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/zones/{profile_id}")
    def save_zone_profile(profile_id: str, payload: dict = Body(...)) -> dict:
        """Create or replace one zone profile."""

        try:
            return ZoneProfileService(Path("zones.json")).save_profile(profile_id, payload)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Single webcam capture ───────────────────────────────────────────────
    @app.post("/run-once")
    def run_once(zone_profile: str | None = Query(default=None)) -> dict:
        pipeline = build_pipeline(zone_profile)
        camera = CameraManager.from_source(pipeline.settings.camera_source)
        try:
            frame = camera.read()
            return pipeline.run(frame)
        finally:
            camera.close()

    # ── Analyze uploaded image (returns full pipeline JSON) ─────────────────
    @app.post("/analyze-image")
    async def analyze_image(file: UploadFile = File(...), zone_profile: str | None = Query(default=None)) -> dict:
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
        return build_pipeline(zone_profile).run(frame)

    # ── Visualize YOLO detections — returns annotated JPEG image ───────────
    @app.post("/visualize-image")
    async def visualize_image(
        file: UploadFile = File(...),
        zone_profile: str | None = Query(default=None),
    ) -> Response:
        """Run YOLO on an uploaded image and return the annotated JPEG with bounding boxes.

        This endpoint is meant for visual debugging: it shows exactly what YOLO
        detects, including class names, confidence scores, and bounding boxes.
        """
        if file.content_type is not None and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        try:
            raw = await file.read()
            image = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        img_array = np.array(image)
        pipeline = build_pipeline(zone_profile)
        detector = pipeline._get_detector()

        # Run YOLO prediction (returns Ultralytics Results list)
        results = detector._model.predict(img_array, conf=detector.conf_threshold, classes=[39])

        # .plot() draws boxes, labels and confidence scores onto a BGR numpy array
        annotated_bgr = results[0].plot()

        # Draw configured zone rectangles on top so you can verify alignment
        annotated_bgr = _draw_zones(annotated_bgr, pipeline.zones)

        # Encode to JPEG and return as image response
        success, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image")

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    # ── Analyze uploaded video ──────────────────────────────────────────────
    @app.post("/analyze-video")
    async def analyze_video(file: UploadFile = File(...), zone_profile: str | None = Query(default=None)) -> dict:
        if file.content_type is not None and not (
            file.content_type.startswith("video/") or file.content_type == "application/octet-stream"
        ):
            raise HTTPException(status_code=400, detail="Only video files are allowed")

        suffix = Path(file.filename or "uploaded-video.mp4").suffix or ".mp4"
        temp_path: Path | None = None
        try:
            raw = await file.read()
            with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(raw)
                temp_path = Path(temp_file.name)

            analyzer = VideoAnalyzer(
                pipeline=build_pipeline(zone_profile),
                config=VideoSamplingConfig(sample_every_seconds=0.0, max_samples=999999),
            )
            return analyzer.analyze(temp_path, file.filename or "uploaded-video")
        except VideoAnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    # ── Analyze uploaded video (Streaming) ──────────────────────────────────
    @app.post("/analyze-video-stream")
    async def analyze_video_stream(
        file: UploadFile = File(...),
        zone_profile: str | None = Query(default=None),
    ) -> StreamingResponse:
        if file.content_type is not None and not (
            file.content_type.startswith("video/") or file.content_type == "application/octet-stream"
        ):
            raise HTTPException(status_code=400, detail="Only video files are allowed")

        suffix = Path(file.filename or "uploaded-video.mp4").suffix or ".mp4"
        
        async def event_generator():
            temp_path: Path | None = None
            try:
                raw = await file.read()
                with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(raw)
                    temp_path = Path(temp_file.name)

                analyzer = VideoAnalyzer(
                    pipeline=build_pipeline(zone_profile),
                    config=VideoSamplingConfig(sample_every_seconds=0.0, max_samples=999999),
                )
                
                # Yield results frame by frame
                for chunk in analyzer.analyze_stream(temp_path, file.filename or "uploaded-video"):
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
            except Exception as exc:
                logger.exception("Error during video streaming")
                error_event = {"type": "error", "error": str(exc)}
                yield f"data: {json.dumps(error_event)}\n\n"
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


# ── helpers ─────────────────────────────────────────────────────────────────

def _draw_zones(image: np.ndarray, zones_config) -> np.ndarray:
    """Overlay configured zone rectangles on the annotated image.

    Each zone is drawn with a distinct color and its name so the user can
    visually confirm that the zone coordinates match the physical shelf layout.
    """
    ZONE_COLORS = {
        "top":    (0, 255, 128),   # green-ish
        "middle": (0, 200, 255),   # cyan
        "bottom": (255, 160, 0),   # orange
    }
    DEFAULT_COLOR = (180, 180, 180)

    height, width = image.shape[:2]
    for zone in zones_config.profile.to_pixel_zones(width, height):
        color = ZONE_COLORS.get(zone.zone_id, DEFAULT_COLOR)
        x1, y1, x2, y2 = int(zone.x1), int(zone.y1), int(zone.x2), int(zone.y2)

        # Semi-transparent rectangle overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
        image = cv2.addWeighted(overlay, 0.08, image, 0.92, 0)

        # Solid border
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # Zone label
        label = zone.name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (x1, y1), (x1 + tw + 8, y1 + th + 8), color, thickness=-1)
        cv2.putText(
            image, label,
            (x1 + 4, y1 + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 2, cv2.LINE_AA,
        )

    return image
